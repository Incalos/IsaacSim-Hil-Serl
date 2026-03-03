from typing import Dict, Optional, Tuple, FrozenSet, Iterable, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.amp import autocast, GradScaler
from serl_launcher.networks.actor_critic_nets import Policy, Critic, CriticEnsemble
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.vision.resnet_v1 import create_encoder
from serl_launcher.common.encoding import EncodingWrapper


class SACAgent:

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        temp: nn.Module,
        encoder: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temp_optimizer: torch.optim.Optimizer,
        encoder_optimizer: torch.optim.Optimizer,
        config: dict,
    ):
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.temp = temp
        self.encoder = encoder
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.temp_optimizer = temp_optimizer
        self.encoder_optimizer = encoder_optimizer
        self.config = config
        self.device = next(actor.parameters()).device
        self._training = True
        self.scaler = GradScaler()

    def state_dict(self) -> dict:
        serializable_config = {k: v for k, v in self.config.items() if not callable(v)}
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "temp": self.temp.state_dict(),
            "encoder": self.encoder.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temp_optimizer": self.temp_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "config": serializable_config,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        self.critic_target.load_state_dict(state_dict["critic_target"], strict=strict)
        self.temp.load_state_dict(state_dict["temp"], strict=strict)
        self.encoder.load_state_dict(state_dict["encoder"], strict=strict)
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        if "critic_optimizer" in state_dict:
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if "temp_optimizer" in state_dict:
            self.temp_optimizer.load_state_dict(state_dict["temp_optimizer"])
        if "encoder_optimizer" in state_dict:
            self.encoder_optimizer.load_state_dict(state_dict["encoder_optimizer"])
        if "config" in state_dict:
            self.config.update(state_dict["config"])

    def to(self, device: torch.device) -> "SACAgent":
        device = torch.device(device) if isinstance(device, str) else device
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.critic_target = self.critic_target.to(device)
        self.temp = self.temp.to(device)
        self.encoder = self.encoder.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True) -> "SACAgent":
        self._training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(False)
        self.temp.train(mode)
        self.encoder.train(mode)
        return self

    def eval(self) -> "SACAgent":
        return self.train(False)

    def _compute_next_actions(self, obs_enc: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample actions and log-probabilities from policy
        next_action_distribution = self.actor(obs_enc)
        next_actions, next_actions_log_probs = next_action_distribution.sample_and_log_prob()
        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, obs_enc: torch.Tensor, next_obs_enc: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # Calculate critic loss with bootstrapped targets and entropy backup
        batch_size = batch["rewards"].shape[0]
        with torch.no_grad():
            next_actions, next_actions_log_probs = self._compute_next_actions(next_obs_enc, batch)
            target_qs = self.critic_target(next_obs_enc, next_actions)
            assert target_qs.shape == (self.config["critic_ensemble_size"], batch_size)

            # Subsample critic ensemble if configured
            if self.config["critic_subsample_size"] is not None:
                indices = torch.randperm(self.config["critic_ensemble_size"], device=self.device)
                indices = indices[:self.config["critic_subsample_size"]]
                target_qs = target_qs[indices]

            # Use min of ensemble Q-values for target (double Q-learning)
            target_q = target_qs.min(dim=0)[0]
            target = batch["rewards"] + self.config["discount"] * batch["masks"] * target_q

            # Add entropy regularization to target if enabled
            if self.config["backup_entropy"]:
                temperature = self.temp()
                target = target - temperature * next_actions_log_probs

        # Compute current Q-values and MSE loss
        current_qs = self.critic(obs_enc, batch["actions"])
        assert current_qs.shape == (self.config["critic_ensemble_size"], batch_size)
        critic_loss = F.mse_loss(current_qs, target.unsqueeze(0).expand(self.config["critic_ensemble_size"], -1))

        # Collect loss metrics
        info = {
            "critic_loss": critic_loss.item(),
            "q_values": current_qs.mean().item(),
            "target_q": target.mean().item(),
        }
        return critic_loss, info

    def actor_loss_fn(self, obs_enc: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Calculate entropy-regularized actor loss
        temperature = self.temp().detach()
        dist = self.actor(obs_enc)
        actions, log_probs = dist.sample_and_log_prob()

        # Use mean of critic ensemble Q-values for policy evaluation
        q_values = self.critic(obs_enc, actions)
        q_values = q_values.mean(dim=0)

        # Actor loss = temperature * log_prob - Q-value (maximize Q - entropy)
        actor_loss = (temperature * log_probs - q_values).mean()

        # Collect loss metrics
        info = {
            "actor_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item(),
            "temperature": temperature.item(),
        }
        return actor_loss, info

    def temperature_loss_fn(self, obs_enc: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # Adjust temperature to match target entropy using Lagrange multiplier
        with torch.no_grad():
            _, log_probs = self._compute_next_actions(obs_enc, batch)
            entropy = -log_probs.mean()

        temperature_loss = self.temp(lhs=entropy, rhs=self.config["target_entropy"])
        info = {"temperature_loss": temperature_loss.item()}
        return temperature_loss, info

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        # Recursively move batch data to agent device
        result = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                result[k] = self._move_batch_to_device(v)
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, np.ndarray):
                result[k] = torch.from_numpy(v).to(self.device, dtype=torch.float32)
            else:
                result[k] = v
        return result

    def update(
            self,
            batch: Dict[str, torch.Tensor],
            networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"}),
    ) -> Dict:
        # Main SAC update step for selected networks
        batch = self._move_batch_to_device(batch)

        # Apply observation augmentation if configured
        if self.config.get("augmentation_function") is not None:
            aug_seed = torch.randint(0, 2**31, (1,)).item()
            batch = self.config["augmentation_function"](batch, aug_seed)

        # Apply reward bias if configured
        reward_bias = self.config.get("reward_bias", 0.0)
        if reward_bias != 0.0:
            batch["rewards"] = batch["rewards"] + reward_bias

        info = {}
        with autocast("cuda"):
            # Encode observations
            obs_enc = self.encoder(batch["observations"])
            next_obs_enc = self.encoder(batch["next_observations"])

            # Update critic and encoder
            if "critic" in networks_to_update:
                self.critic_optimizer.zero_grad()
                self.encoder_optimizer.zero_grad()
                critic_loss, critic_info = self.critic_loss_fn(obs_enc, next_obs_enc.detach(), batch)
                self.scaler.scale(critic_loss).backward()
                self.scaler.step(self.critic_optimizer)
                self.scaler.step(self.encoder_optimizer)
                info.update(critic_info)

                # Soft update critic target network
                with torch.no_grad():
                    tau = self.config["soft_target_update_rate"]
                    for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target.data.lerp_(source.data, tau)

            # Update actor
            if "actor" in networks_to_update:
                self.actor_optimizer.zero_grad()
                actor_loss, actor_info = self.actor_loss_fn(obs_enc.detach())
                self.scaler.scale(actor_loss).backward()
                self.scaler.step(self.actor_optimizer)
                info.update(actor_info)

            # Update temperature
            if "temperature" in networks_to_update:
                self.temp_optimizer.zero_grad()
                temp_loss, temp_info = self.temperature_loss_fn(obs_enc.detach(), batch)
                self.scaler.scale(temp_loss).backward()
                self.scaler.step(self.temp_optimizer)
                info.update(temp_info)

        # Update gradient scaler for mixed precision
        self.scaler.update()
        return info

    @torch.no_grad()
    def sample_actions(self, observations: Dict[str, torch.Tensor], argmax: bool = False) -> torch.Tensor:
        # Sample actions from policy (argmax for deterministic inference)
        observations = self._move_batch_to_device(observations)
        obs_enc = self.encoder(observations)
        dist = self.actor(obs_enc)
        return dist.mode() if argmax else dist.sample()

    @classmethod
    def create_pixels(
            cls,
            sample_obs: Dict[str, torch.Tensor],
            sample_action: torch.Tensor,
            encoder_type: str = "resnet18-pretrained",
            use_proprio: bool = False,
            critic_network_kwargs: dict = None,
            policy_network_kwargs: dict = None,
            policy_kwargs: dict = None,
            critic_ensemble_size: int = 2,
            critic_subsample_size: Optional[int] = None,
            temperature_init: float = 1e-2,
            image_keys: Iterable[str] = ("image",),
            augmentation_function: Optional[Callable] = None,
            reward_bias: float = 0.0,
            image_size: Tuple[int, int] = (128, 128),
            **kwargs,
    ) -> "SACAgent":
        # Factory method to create SACAgent for pixel observations
        image_keys = tuple(image_keys)
        action_dim = sample_action.shape[-1]

        # Default network hyperparameters
        critic_network_kwargs = critic_network_kwargs or {"hidden_dims": [256, 256]}
        policy_network_kwargs = policy_network_kwargs or {"hidden_dims": [256, 256]}
        policy_kwargs = policy_kwargs or {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        }
        policy_network_kwargs.setdefault("activate_final", True)
        critic_network_kwargs.setdefault("activate_final", True)

        # Create visual encoder
        encoders = create_encoder(
            encoder_type=encoder_type,
            image_keys=image_keys,
            image_size=image_size,
            pooling_method="avg",
            num_spatial_blocks=8,
            bottleneck_dim=256,
        )
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            proprio_latent_dim=64,
            enable_stacking=True,
            image_keys=image_keys,
        )

        # Determine encoder output dimension
        dummy_obs = {k: torch.zeros(1, *v.shape[1:], dtype=torch.float32) for k, v in sample_obs.items() if hasattr(v, "shape")}
        with torch.no_grad():
            _ = encoder_def(dummy_obs, train=False)
        encoder_output_dim = encoder_def.output_dim

        # Build policy network
        policy_hidden_dims = [encoder_output_dim] + policy_network_kwargs["hidden_dims"]
        actor = Policy(
            network=MLP(
                hidden_dims=policy_hidden_dims,
                activate_final=True,
                use_layer_norm=policy_network_kwargs.get("use_layer_norm", False),
                activations=policy_network_kwargs.get("activation", nn.SiLU()),
            ),
            action_dim=action_dim,
            **policy_kwargs,
        )

        # Build critic ensemble
        critic_hidden_dims = [encoder_output_dim + action_dim] + critic_network_kwargs["hidden_dims"]
        critics = [
            Critic(
                MLP(
                    hidden_dims=critic_hidden_dims,
                    activate_final=True,
                    use_layer_norm=critic_network_kwargs.get("use_layer_norm", False),
                    activations=critic_network_kwargs.get("activation", nn.SiLU()),
                )) for _ in range(critic_ensemble_size)
        ]
        critic = CriticEnsemble(critics)
        critic_target = deepcopy(critic)

        # Initialize temperature (alpha) parameter
        temp = GeqLagrangeMultiplier(init_value=temperature_init, constraint_shape=())

        # Configure agent hyperparameters
        target_entropy = kwargs.get("target_entropy", -float(action_dim))
        config_kwargs = {
            "discount": kwargs.get("discount", 0.97),
            "soft_target_update_rate": kwargs.get("soft_target_update_rate", 0.005),
            "target_entropy": target_entropy,
            "backup_entropy": kwargs.get("backup_entropy", True),
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            "image_keys": image_keys,
            "augmentation_function": augmentation_function,
            "reward_bias": reward_bias,
        }

        # Create SAC agent instance
        agent = cls(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            temp=temp,
            encoder=encoder_def,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
            critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
            temp_optimizer=torch.optim.Adam(temp.parameters(), lr=3e-4),
            encoder_optimizer=torch.optim.Adam(encoder_def.parameters(), lr=3e-4),
            config=config_kwargs,
        )
        return agent

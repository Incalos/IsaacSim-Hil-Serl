import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, FrozenSet, Iterable, Callable
from copy import deepcopy
from torch.amp import autocast, GradScaler
from serl_launcher.networks.actor_critic_nets import Policy, Critic, CriticEnsemble
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.vision.resnet_v1 import create_encoder
from serl_launcher.common.encoding import EncodingWrapper


class SACAgent:
    """Soft Actor-Critic (SAC) agent with visual encoder support for pixel-based observations.

    Args:
        actor: Policy network for action generation
        critic: Critic ensemble network for Q-value estimation
        critic_target: Target critic network for stable TD learning
        temp: Temperature (Lagrange multiplier) for entropy regularization
        encoder: Visual/proprioceptive encoder network
        actor_optimizer: Optimizer for actor network parameters
        critic_optimizer: Optimizer for critic network parameters
        temp_optimizer: Optimizer for temperature parameter
        encoder_optimizer: Optimizer for encoder network parameters
        config: Dictionary of agent configuration parameters
    """

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
        """Get serializable state dictionary containing network and optimizer states.

        Returns:
            Dictionary with network parameters, optimizer states, and serializable config
        """
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
        """Load network and optimizer states from state dictionary.

        Args:
            state_dict: Dictionary containing saved agent state
            strict: Whether to enforce strict weight loading (default: True)
        """
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
        """Move agent networks to specified device.

        Args:
            device: Target device (torch.device or string)

        Returns:
            Self for method chaining
        """
        device = torch.device(device) if isinstance(device, str) else device

        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.critic_target = self.critic_target.to(device)
        self.temp = self.temp.to(device)
        self.encoder = self.encoder.to(device)

        self.device = device
        return self

    def train(self, mode: bool = True) -> "SACAgent":
        """Set agent training/evaluation mode.

        Args:
            mode: True for training mode, False for evaluation (default: True)

        Returns:
            Self for method chaining
        """
        self._training = mode

        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(False)
        self.temp.train(mode)
        self.encoder.train(mode)

        return self

    def eval(self) -> "SACAgent":
        """Set agent to evaluation mode (wrapper for train(False)).

        Returns:
            Self for method chaining
        """
        return self.train(False)

    def _compute_next_actions(
        self,
        obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute next actions and their log probabilities from actor network.

        Args:
            obs_enc: Encoded next observations (shape: [batch_size, encoder_dim])
            batch: Dictionary containing batch data with "actions" key

        Returns:
            Tuple of (next_actions, next_actions_log_probs) where:
                next_actions: Sampled actions (shape: [batch_size, action_dim])
                next_actions_log_probs: Log probabilities of sampled actions (shape: [batch_size])
        """
        next_action_distribution = self.actor(obs_enc)
        next_actions, next_actions_log_probs = next_action_distribution.sample_and_log_prob()

        assert next_actions.shape == batch["actions"].shape
        assert next_actions_log_probs.shape == (batch["actions"].shape[0],)

        return next_actions, next_actions_log_probs

    def critic_loss_fn(
        self,
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute critic loss using TD error with target Q-values.

        Args:
            obs_enc: Encoded current observations (shape: [batch_size, encoder_dim])
            next_obs_enc: Encoded next observations (shape: [batch_size, encoder_dim])
            batch: Dictionary containing batch data (rewards, masks, actions)

        Returns:
            Tuple of (critic_loss, info) where:
                critic_loss: Mean squared error loss between current and target Q-values
                info: Dictionary with logging metrics (critic_loss, q_values, target_q)
        """
        batch_size = batch["rewards"].shape[0]

        with torch.no_grad():
            next_actions, next_actions_log_probs = self._compute_next_actions(next_obs_enc, batch)
            target_qs = self.critic_target(next_obs_enc, next_actions)

            if self.config["critic_subsample_size"] is not None:
                indices = torch.randperm(self.config["critic_ensemble_size"])
                indices = indices[: self.config["critic_subsample_size"]]
                target_qs = target_qs[indices]

            target_q = target_qs.min(dim=0)[0]
            assert target_q.shape == (batch_size,)

            target = batch["rewards"] + self.config["discount"] * batch["masks"] * target_q

            if self.config["backup_entropy"]:
                temperature = self.temp()
                target = target - temperature * next_actions_log_probs

        current_qs = self.critic(obs_enc, batch["actions"])
        assert current_qs.shape == (self.config["critic_ensemble_size"], batch_size)

        critic_loss = F.mse_loss(current_qs, target.unsqueeze(0).expand(self.config["critic_ensemble_size"], -1))

        info = {
            "critic_loss": critic_loss.item(),
            "q_values": current_qs.mean().item(),
            "target_q": target.mean().item(),
        }

        return critic_loss, info

    def actor_loss_fn(self, obs_enc: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute actor loss with entropy regularization.

        Args:
            obs_enc: Encoded observations (shape: [batch_size, encoder_dim])

        Returns:
            Tuple of (actor_loss, info) where:
                actor_loss: Actor loss (temperature * log_probs - Q_values) mean
                info: Dictionary with logging metrics (actor_loss, entropy, temperature)
        """
        temperature = self.temp().detach()
        dist = self.actor(obs_enc)
        actions, log_probs = dist.sample_and_log_prob()

        q_values = self.critic(obs_enc, actions)
        q_values = q_values.mean(dim=0)

        actor_loss = (temperature * log_probs - q_values).mean()

        info = {
            "actor_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item(),
            "temperature": temperature.item(),
        }

        return actor_loss, info

    def temperature_loss_fn(self, obs_enc: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute temperature loss for entropy constraint satisfaction.

        Args:
            obs_enc: Encoded next observations (shape: [batch_size, encoder_dim])
            batch: Dictionary containing batch data

        Returns:
            Tuple of (temperature_loss, info) where:
                temperature_loss: Lagrange multiplier loss for entropy constraint
                info: Dictionary with logging metric (temperature_loss)
        """
        _, next_actions_log_probs = self._compute_next_actions(obs_enc, batch)
        entropy = -next_actions_log_probs.mean()

        temperature_loss = self.temp(lhs=entropy.detach(), rhs=self.config["target_entropy"])

        info = {"temperature_loss": temperature_loss.item()}
        return temperature_loss, info

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Recursively move batch tensors to agent's device.

        Args:
            batch: Dictionary of batch data (tensors, numpy arrays, nested dicts)

        Returns:
            Dictionary with all tensor data moved to agent's device
        """
        result = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                result[k] = self._move_batch_to_device(v)
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, np.ndarray):
                result[k] = torch.from_numpy(v).to(self.device)
            else:
                result[k] = v
        return result

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"}),
    ) -> Dict:
        """Update agent networks (critic/actor/temperature) using batch data.

        Args:
            batch: Dictionary containing batch data (observations, next_observations, rewards, etc.)
            networks_to_update: Set of networks to update (default: {"actor", "critic", "temperature"})

        Returns:
            Dictionary with logging metrics from network updates
        """
        batch = self._move_batch_to_device(batch)

        if self.config.get("augmentation_function") is not None:
            aug_seed = torch.randint(0, 2**31, (1,)).item()
            batch = self.config["augmentation_function"](batch, aug_seed)

        reward_bias = self.config.get("reward_bias", 0.0)
        if reward_bias != 0.0:
            batch = {**batch, "rewards": batch["rewards"] + reward_bias}

        info = {}

        obs_enc = self.encoder(batch["observations"])
        next_obs_enc = self.encoder(batch["next_observations"])

        if "critic" in networks_to_update:
            self.critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()

            with autocast("cuda"):
                critic_loss, critic_info = self.critic_loss_fn(obs_enc, next_obs_enc.detach(), batch)

            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
            self.scaler.step(self.encoder_optimizer)
            self.scaler.update()

            info.update(critic_info)

            with torch.no_grad():
                tau = self.config["soft_target_update_rate"]
                for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target.data.mul_(1 - tau)
                    target.data.add_(tau * source.data)

        if "actor" in networks_to_update:
            self.actor_optimizer.zero_grad()

            with autocast("cuda"):
                actor_loss, actor_info = self.actor_loss_fn(obs_enc.detach())

            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()

            info.update(actor_info)

        if "temperature" in networks_to_update:
            self.temp_optimizer.zero_grad()

            with autocast("cuda"):
                temp_loss, temp_info = self.temperature_loss_fn(next_obs_enc.detach(), batch)

            self.scaler.scale(temp_loss).backward()
            self.scaler.step(self.temp_optimizer)
            self.scaler.update()

            info.update(temp_info)

        return info

    @torch.no_grad()
    def sample_actions(self, observations: Dict[str, torch.Tensor], argmax: bool = False) -> torch.Tensor:
        """Sample actions from actor network (or get argmax for evaluation).

        Args:
            observations: Dictionary of observation tensors (pixel/proprio)
            argmax: Whether to return argmax action (True) or sampled action (False) (default: False)

        Returns:
            Action tensor (shape: [batch_size, action_dim])
        """
        observations = self._move_batch_to_device(observations)
        obs_enc = self.encoder(observations)
        dist = self.actor(obs_enc)

        if argmax:
            return dist.mode()
        return dist.sample()

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
        """Create SACAgent instance for pixel-based observations.

        Args:
            sample_obs: Sample observation dictionary to infer input shapes
            sample_action: Sample action tensor to infer action dimension
            encoder_type: Type of visual encoder (default: "resnet18-pretrained")
            use_proprio: Whether to use proprioceptive features (default: False)
            critic_network_kwargs: Keyword arguments for critic MLP (default: None)
            policy_network_kwargs: Keyword arguments for policy MLP (default: None)
            policy_kwargs: Keyword arguments for Policy class (default: None)
            critic_ensemble_size: Number of critics in ensemble (default: 2)
            critic_subsample_size: Number of critics to subsample for target Q (default: None)
            temperature_init: Initial temperature value (default: 1e-2)
            image_keys: Iterable of keys for image observations (default: ("image",))
            augmentation_function: Function for data augmentation (default: None)
            reward_bias: Constant bias added to rewards (default: 0.0)
            image_size: Image dimensions (height, width) (default: (128, 128))
            **kwargs: Additional keyword arguments (discount, target_entropy, etc.)

        Returns:
            Initialized SACAgent instance for pixel observations
        """
        image_keys = tuple(image_keys)

        if critic_network_kwargs is None:
            critic_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_network_kwargs is None:
            policy_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_kwargs is None:
            policy_kwargs = {
                "tanh_squash_distribution": True,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
            }

        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}
        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}

        action_dim = sample_action.shape[-1]

        encoders = create_encoder(
            encoder_type=encoder_type,
            image_keys=image_keys,
            image_size=image_size,
            pooling_method="spatial_learned_embeddings",
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

        dummy_obs = {}
        for k, v in sample_obs.items():
            if isinstance(v, torch.Tensor):
                dummy_obs[k] = torch.zeros(1, *v.shape[1:], device="cpu", dtype=v.dtype)
            elif isinstance(v, np.ndarray):
                dummy_obs[k] = torch.zeros(1, *v.shape[1:], device="cpu", dtype=torch.float32)
            else:
                dummy_obs[k] = v
        with torch.no_grad():
            _ = encoder_def(dummy_obs, train=False)
        encoder_output_dim = encoder_def.output_dim

        policy_hidden_dims = [encoder_output_dim] + policy_network_kwargs.get("hidden_dims", [256, 256])
        policy_network = MLP(
            hidden_dims=policy_hidden_dims,
            activate_final=True,
            use_layer_norm=policy_network_kwargs.get("use_layer_norm", False),
            activations=policy_network_kwargs.get("activation", nn.SiLU()),
        )
        actor = Policy(
            network=policy_network,
            action_dim=action_dim,
            **policy_kwargs,
        )

        critic_hidden_dims = [encoder_output_dim + action_dim] + critic_network_kwargs.get("hidden_dims", [256, 256])
        critics = []
        for _ in range(critic_ensemble_size):
            critic_network = MLP(
                hidden_dims=critic_hidden_dims,
                activate_final=True,
                use_layer_norm=critic_network_kwargs.get("use_layer_norm", False),
                activations=critic_network_kwargs.get("activation", nn.SiLU()),
            )
            critics.append(Critic(network=critic_network))
        critic = CriticEnsemble(critics)
        critic_target = deepcopy(critic)

        temp = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )

        target_entropy = kwargs.get("target_entropy")
        if target_entropy is None:
            target_entropy = -action_dim / 2

        config_kwargs = {
            "discount": kwargs.get("discount", 0.97),
            "soft_target_update_rate": kwargs.get("soft_target_update_rate", 0.005),
            "target_entropy": target_entropy,
            "backup_entropy": kwargs.get("backup_entropy", False),
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            "image_keys": image_keys,
            "augmentation_function": augmentation_function,
            "reward_bias": reward_bias,
        }

        temp_optimizer = torch.optim.Adam(temp.parameters(), lr=3e-4)
        encoder_optimizer = torch.optim.Adam(encoder_def.parameters(), lr=3e-4)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

        agent = cls(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            temp=temp,
            encoder=encoder_def,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temp_optimizer=temp_optimizer,
            encoder_optimizer=encoder_optimizer,
            config=config_kwargs,
        )

        return agent

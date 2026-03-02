from typing import Optional, Sequence, Dict, Union
import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from serl_launcher.networks.mlp import default_init


class TanhNormal(TransformedDistribution):

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__(Normal(loc, scale), [TanhTransform()])

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.base_dist.loc)

    def sample_and_log_prob(self, sample_shape=torch.Size()):
        samples = self.rsample(sample_shape)
        log_probs = self.log_prob(samples)
        return samples, log_probs.sum(dim=-1)


class ValueCritic(nn.Module):

    def __init__(self, encoder: nn.Module, network: nn.Module, init_final: Optional[float] = None):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final

        if init_final is not None:
            self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            default_init()(self.output_layer.weight)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Encode observations and pass through network
        x = self.network(self.encoder(observations, train))
        value = self.output_layer(x)
        return value.squeeze(-1)


class Critic(nn.Module):

    def __init__(self, network: nn.Module, init_final: Optional[float] = None):
        super().__init__()
        self.network = network
        self.init_final = init_final

        if init_final is not None:
            self.output_layer = nn.Linear(network.out_dim, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.out_dim, 1)
            default_init()(self.output_layer.weight)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Concatenate observations and actions as input
        inputs = torch.cat([observations, actions], dim=-1)
        x = self.network(inputs)
        value = self.output_layer(x)
        return value.squeeze(-1)

    def q_value_ensemble(self, observations: torch.Tensor, actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Handle 3D action tensor (batch x num_actions x action_dim)
        if len(actions.shape) == 3:
            batch_size, num_actions = actions.shape[:2]
            # Expand observations to match action batch size
            obs_expanded = observations.unsqueeze(1).expand(-1, num_actions, -1)
            obs_flat = obs_expanded.reshape(-1, observations.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])

            # Compute Q-values for flattened inputs
            q_values = self(obs_flat, actions_flat, train)
            return q_values.reshape(batch_size, num_actions)
        else:
            return self(observations, actions, train)


class GraspCritic(nn.Module):

    def __init__(self, network: nn.Module, init_final: Optional[float] = None, output_dim: int = 3):
        super().__init__()
        self.network = network
        self.init_final = init_final
        self.output_dim = output_dim

        if init_final is not None:
            self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            default_init()(self.output_layer.weight)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Predict grasp parameters from observations
        x = self.network(observations)
        return self.output_layer(x)


class Policy(nn.Module):

    def __init__(
        self,
        network: nn.Module,
        action_dim: int,
        std_parameterization: str = "exp",
        std_min: float = 1e-5,
        std_max: float = 10.0,
        tanh_squash_distribution: bool = False,
        fixed_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.network = network
        self.action_dim = action_dim
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std

        # Initialize mean prediction layer
        self.mean_layer = nn.Linear(network.out_dim, action_dim)
        default_init()(self.mean_layer.weight)

        # Initialize std layer if not using fixed std
        if fixed_std is None:
            self.std_layer = nn.Linear(network.out_dim, action_dim)
            default_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        temperature: float = 1.0,
        train: bool = False,
        non_squash_distribution: bool = False,
    ) -> TransformedDistribution:
        # Extract features from observations
        features = self.network(observations)
        means = self.mean_layer(features)

        # Calculate standard deviation based on parameterization
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = self.std_layer(features)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.functional.softplus(self.std_layer(features))
            elif self.std_parameterization == "uniform":
                log_stds = self.std_layer.bias
                stds = torch.exp(log_stds).expand_as(means)
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            assert self.std_parameterization == "fixed"
            stds = self.fixed_std.expand_as(means)

        # Clamp std values and apply temperature scaling
        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(torch.tensor(temperature, device=stds.device))

        # Handle NaN values for stability
        if torch.isnan(means).any():
            means = torch.nan_to_num(means, nan=0.0)
        if torch.isnan(stds).any():
            stds = torch.nan_to_num(stds, nan=self.std_min)

        # Return appropriate distribution (tanh-squashed or normal)
        if self.tanh_squash_distribution and not non_squash_distribution:
            return TanhNormal(means, stds)
        else:
            return Normal(means, stds)


def create_critic_ensemble(critic_class, num_critics: int) -> nn.ModuleList:
    # Create list of critic instances
    return nn.ModuleList([critic_class() for _ in range(num_critics)])


class CriticEnsemble(nn.Module):

    def __init__(self, critics: Sequence[nn.Module]):
        super().__init__()
        self.critics = nn.ModuleList(critics)
        self.num_critics = len(critics)

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]], actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Compute Q-values from all critics and stack results
        q_values = [critic(observations, actions, train) for critic in self.critics]
        return torch.stack(q_values, dim=0)

    def q_min(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]], actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Get minimum Q-value across the ensemble
        q_values = self(observations, actions, train)
        return q_values.min(dim=0)[0]

    def q_mean(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]], actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Get mean Q-value across the ensemble
        q_values = self(observations, actions, train)
        return q_values.mean(dim=0)

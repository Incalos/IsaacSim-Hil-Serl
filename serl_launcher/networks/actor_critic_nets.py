from typing import Optional, Sequence, Dict, Union
import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from serl_launcher.networks.mlp import default_init


class TanhNormal(TransformedDistribution):
    """Tanh-squashed normal distribution for bounded action spaces.

    Wraps a Normal distribution with Tanh transform to constrain outputs to [-1, 1].

    Args:
        loc: Mean of the base normal distribution
        scale: Standard deviation of the base normal distribution
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__(Normal(loc, scale), [TanhTransform()])

    def mode(self) -> torch.Tensor:
        """Return mode of the tanh-normal distribution (tanh of mean).

        Returns:
            Mode tensor (same shape as loc/scale)
        """
        return torch.tanh(self.base_dist.loc)

    def sample_and_log_prob(self, sample_shape=torch.Size()):
        """Sample from distribution and compute log probability.

        Args:
            sample_shape: Shape of samples to generate (default: torch.Size())

        Returns:
            Tuple of (samples, log probabilities summed over last dimension)
        """
        samples = self.rsample(sample_shape)
        log_probs = self.log_prob(samples)
        return samples, log_probs.sum(dim=-1)


class ValueCritic(nn.Module):
    """Value critic network (state -> value scalar).

    Args:
        encoder: Encoder module to process raw observations
        network: Core network (MLP/ResNet) for feature processing
        init_final: Range for uniform initialization of output layer (None for orthogonal init) (default: None)
    """

    def __init__(self, encoder: nn.Module, network: nn.Module, init_final: Optional[float] = None):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final

        self.output_layer = nn.Linear(network.net[-2].out_features, 1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            default_init()(self.output_layer.weight)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward pass to predict state value.

        Args:
            observations: Input observation tensor (shape: [batch_size, obs_dim])
            train: Whether to set encoder/network to training mode (default: False)

        Returns:
            Value scalar tensor (shape: [batch_size])
        """
        x = self.network(self.encoder(observations, train))
        value = self.output_layer(x)
        return value.squeeze(-1)


class Critic(nn.Module):
    """Q-value critic network (state + action -> Q scalar).

    Args:
        network: Core network (MLP/ResNet) for feature processing
        init_final: Range for uniform initialization of output layer (None for orthogonal init) (default: None)
    """

    def __init__(self, network: nn.Module, init_final: Optional[float] = None):
        super().__init__()
        self.network = network
        self.init_final = init_final

        self.output_layer = nn.Linear(network.out_dim, 1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            default_init()(self.output_layer.weight)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward pass to predict Q-value for state-action pair.

        Args:
            observations: State tensor (shape: [batch_size, obs_dim])
            actions: Action tensor (shape: [batch_size, action_dim])
            train: Whether to set network to training mode (default: False)

        Returns:
            Q-value scalar tensor (shape: [batch_size])
        """
        inputs = torch.cat([observations, actions], dim=-1)
        x = self.network(inputs)
        value = self.output_layer(x)
        return value.squeeze(-1)

    def q_value_ensemble(self, observations: torch.Tensor, actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Compute Q-values for batch of state-action ensembles.

        Handles batched actions with shape [Batch, num_actions, action_dim].

        Args:
            observations: State tensor (shape: [batch_size, obs_dim])
            actions: Action tensor (shape: [batch_size, num_actions, action_dim] or [batch_size, action_dim])
            train: Whether to set network to training mode (default: False)

        Returns:
            Q-values tensor (shape: [batch_size, num_actions] or [batch_size])
        """
        if len(actions.shape) == 3:
            batch_size, num_actions = actions.shape[:2]
            obs_expanded = observations.unsqueeze(1).expand(-1, num_actions, -1)
            obs_flat = obs_expanded.reshape(-1, observations.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
            q_values = self(obs_flat, actions_flat, train)
            return q_values.reshape(batch_size, num_actions)
        else:
            return self(observations, actions, train)


class GraspCritic(nn.Module):
    """Grasp critic network (state -> grasp parameters).

    Args:
        network: Core network (MLP/ResNet) for feature processing
        init_final: Range for uniform initialization of output layer (None for orthogonal init) (default: None)
        output_dim: Dimension of grasp parameter output (default: 3)
    """

    def __init__(self, network: nn.Module, init_final: Optional[float] = None, output_dim: int = 3):
        super().__init__()
        self.network = network
        self.init_final = init_final
        self.output_dim = output_dim

        self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            default_init()(self.output_layer.weight)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward pass to predict grasp parameters.

        Args:
            observations: Input observation tensor (shape: [batch_size, obs_dim])
            train: Whether to set network to training mode (default: False)

        Returns:
            Grasp parameters tensor (shape: [batch_size, output_dim])
        """
        x = self.network(observations)
        return self.output_layer(x)


class Policy(nn.Module):
    """Stochastic policy network (state -> action distribution).

    Supports normal/tanh-normal distributions with configurable std parameterization.

    Args:
        network: Core network (MLP/ResNet) for feature processing
        action_dim: Dimension of action space
        std_parameterization: Method to parameterize standard deviation ("exp", "softplus", "uniform", "fixed") (default: "exp")
        std_min: Minimum allowed standard deviation (default: 1e-5)
        std_max: Maximum allowed standard deviation (default: 10.0)
        tanh_squash_distribution: Whether to apply tanh transform to actions (default: False)
        fixed_std: Fixed standard deviation tensor (None for learned std) (default: None)
    """

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

        self.mean_layer = nn.Linear(network.out_dim, action_dim)
        default_init()(self.mean_layer.weight)

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
        """Forward pass to compute action distribution.

        Args:
            observations: Input observation tensor (shape: [batch_size, obs_dim])
            temperature: Temperature scaling for std (higher = more exploration) (default: 1.0)
            train: Whether to set network to training mode (default: False)
            non_squash_distribution: Skip tanh transform (default: False)

        Returns:
            Action distribution (Normal or TanhNormal)

        Raises:
            ValueError: If invalid std parameterization is provided
        """
        features = self.network(observations)
        means = self.mean_layer(features)

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

        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(torch.tensor(temperature, device=stds.device))

        if torch.isnan(means).any():
            means = torch.nan_to_num(means, nan=0.0)
        if torch.isnan(stds).any():
            stds = torch.nan_to_num(stds, nan=self.std_min)

        if self.tanh_squash_distribution and not non_squash_distribution:
            return TanhNormal(means, stds)
        else:
            return Normal(means, stds)


def create_critic_ensemble(critic_class, num_critics: int) -> nn.ModuleList:
    """Create ensemble of critic networks.

    Args:
        critic_class: Critic class to instantiate
        num_critics: Number of critics in ensemble

    Returns:
        ModuleList of initialized critic networks
    """
    return nn.ModuleList([critic_class() for _ in range(num_critics)])


class CriticEnsemble(nn.Module):
    """Ensemble of critic networks for Q-value prediction.

    Args:
        critics: Sequence of pre-initialized critic networks
    """

    def __init__(self, critics: Sequence[nn.Module]):
        super().__init__()
        self.critics = nn.ModuleList(critics)
        self.num_critics = len(critics)

    def forward(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        """Forward pass to compute Q-values from all ensemble members.

        Args:
            observations: State tensor/dict (shape: [batch_size, obs_dim])
            actions: Action tensor (shape: [batch_size, action_dim])
            train: Whether to set critics to training mode (default: False)

        Returns:
            Q-values tensor (shape: [num_critics, batch_size])
        """
        q_values = [critic(observations, actions, train) for critic in self.critics]
        return torch.stack(q_values, dim=0)

    def q_min(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        """Compute minimum Q-value across ensemble (for conservative Q-learning).

        Args:
            observations: State tensor/dict (shape: [batch_size, obs_dim])
            actions: Action tensor (shape: [batch_size, action_dim])
            train: Whether to set critics to training mode (default: False)

        Returns:
            Minimum Q-value tensor (shape: [batch_size])
        """
        q_values = self(observations, actions, train)
        return q_values.min(dim=0)[0]

    def q_mean(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        """Compute mean Q-value across ensemble.

        Args:
            observations: State tensor/dict (shape: [batch_size, obs_dim])
            actions: Action tensor (shape: [batch_size, action_dim])
            train: Whether to set critics to training mode (default: False)

        Returns:
            Mean Q-value tensor (shape: [batch_size])
        """
        q_values = self(observations, actions, train)
        return q_values.mean(dim=0)

from typing import Callable, Optional, Sequence, Union
import torch
import torch.nn as nn


def default_init() -> Callable[[torch.Tensor], torch.Tensor]:
    """Return orthogonal initialization function with gain=1.0.

    Returns:
        Function to apply orthogonal initialization to tensor
    """
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable layers/activations/norm/dropout.

    Args:
        hidden_dims: Sequence of dimensions for MLP layers (input -> hidden -> output)
        activations: Activation function (module or string name) (default: nn.SiLU())
        activate_final: Whether to apply activation to final layer (default: False)
        use_layer_norm: Whether to apply layer normalization (default: False)
        dropout_rate: Dropout probability (None for no dropout) (default: None)
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Union[Callable[[torch.Tensor], torch.Tensor], str] = nn.SiLU(),
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.activate_final = activate_final
        layers = []

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            default_init()(layers[-1].weight)

            if i + 2 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dims[i + 1]))
                layers.append(activations if isinstance(activations, nn.Module) else getattr(nn, activations)())

        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward pass of MLP.

        Args:
            x: Input tensor (shape: [batch_size, input_dim])
            train: Unused (for API consistency) (default: False)

        Returns:
            Output tensor (shape: [batch_size, out_dim])
        """
        return self.net(x)


class MLPResNetBlock(nn.Module):
    """Residual block for MLPResNet with 4x expansion.

    Args:
        features: Dimension of input/output features
        act: Activation function
        dropout_rate: Dropout probability (None for no dropout) (default: None)
        use_layer_norm: Whether to apply layer normalization (default: False)
    """

    def __init__(self, features: int, act: Callable, dropout_rate: float = None, use_layer_norm: bool = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.dense1 = nn.Linear(features, features * 4)
        self.act = act
        self.dense2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward pass of residual block with residual connection.

        Args:
            x: Input tensor (shape: [batch_size, features])
            train: Unused (for API consistency) (default: False)

        Returns:
            Output tensor (shape: [batch_size, features])
        """
        residual = x

        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x


class MLPResNet(nn.Module):
    """ResNet-style MLP with multiple residual blocks.

    Args:
        num_blocks: Number of residual blocks
        out_dim: Dimension of final output
        dropout_rate: Dropout probability (None for no dropout) (default: None)
        use_layer_norm: Whether to apply layer normalization (default: False)
        hidden_dim: Dimension of hidden layers (default: 256)
        activations: Activation function (default: nn.SiLU())
    """

    def __init__(
        self,
        num_blocks: int,
        out_dim: int,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: Callable = nn.SiLU(),
    ):
        super().__init__()
        self.input_layer = nn.Linear(hidden_dim, hidden_dim)
        default_init()(self.input_layer.weight)

        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(hidden_dim, act=activations, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.activations = activations
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        default_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward pass of ResNet-style MLP.

        Args:
            x: Input tensor (shape: [batch_size, hidden_dim])
            train: Unused (for API consistency) (default: False)

        Returns:
            Output tensor (shape: [batch_size, out_dim])
        """
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x, train=train)

        x = self.activations(x)
        x = self.output_layer(x)
        return x


class Scalar(nn.Module):
    """Learnable scalar parameter module.

    Args:
        init_value: Initial value for the scalar parameter
    """

    def __init__(self, init_value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(init_value))

    def forward(self) -> torch.Tensor:
        """Return the learnable scalar value.

        Returns:
            Scalar tensor (shape: [])
        """
        return self.value

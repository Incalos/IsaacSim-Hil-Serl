from functools import partial
from typing import Optional, Sequence
import torch
import torch.nn as nn


class LagrangeMultiplier(nn.Module):
    """Lagrange multiplier module for constraint optimization.

    Supports equality (eq), greater-or-equal (geq), and less-or-equal (leq) constraints.
    Applies parameterization (softplus/exp) for non-negativity of inequality multipliers.

    Args:
        init_value: Initial value for multiplier parameter (default: 1.0)
        constraint_shape: Shape of the lagrange multiplier tensor (default: ())
        constraint_type: Type of constraint ("eq", "leq", "geq") (default: "eq")
        parameterization: Parameterization for inequality constraints ("softplus", "exp", None) (default: None)
    """

    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Sequence[int] = (),
        constraint_type: str = "eq",
        parameterization: Optional[str] = None,
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.parameterization = parameterization

        if constraint_type != "eq":
            assert init_value > 0, "Inequality constraints require positive initial multiplier values"
            if parameterization == "softplus":
                init_value = torch.log(torch.exp(torch.tensor(init_value)) - 1)
            elif parameterization == "exp":
                init_value = torch.log(torch.tensor(init_value))
            elif parameterization != "none":
                raise ValueError(f"Invalid parameterization {parameterization} for inequality constraint")
        else:
            assert parameterization is None, "Equality constraints cannot have parameterization"

        self.lagrange = nn.Parameter(torch.full(constraint_shape, init_value, dtype=torch.float32))

    def forward(self, lhs: Optional[torch.Tensor] = None, rhs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute lagrange penalty or return raw multiplier.

        Args:
            lhs: Left-hand side tensor of constraint (default: None)
            rhs: Right-hand side tensor of constraint (default: None)

        Returns:
            Raw multiplier tensor (if lhs is None) or penalty tensor for constraint violation

        Raises:
            ValueError: If invalid constraint type/parameterization is provided
            AssertionError: If shape mismatch between multiplier and constraint difference
        """
        multiplier = self.lagrange

        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = torch.nn.functional.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = torch.exp(multiplier)
            elif self.parameterization != "none":
                raise ValueError(f"Invalid parameterization {self.parameterization}")

        if lhs is None:
            return multiplier

        if rhs is None:
            rhs = torch.zeros_like(lhs)

        diff = lhs - rhs
        assert diff.shape == multiplier.shape, f"Shape mismatch: {diff.shape} vs {multiplier.shape}"

        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff
        else:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")


GeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="geq", parameterization="softplus")
LeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="leq", parameterization="softplus")
BetterLeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="leq", parameterization="none")

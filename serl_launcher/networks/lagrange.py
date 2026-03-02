from functools import partial
from typing import Optional, Sequence
import torch
import torch.nn as nn


class LagrangeMultiplier(nn.Module):

    def __init__(
            self,
            init_value: float = 1.0,
            constraint_shape: Sequence[int] = (),
            constraint_type: str = "eq",  # Constraint type: "eq" (equality), "leq" (less equal), "geq" (greater equal)
            parameterization: Optional[str] = None,  # Parameterization for inequality constraints: "softplus", "exp", None
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.parameterization = parameterization

        # Validate input for inequality constraints
        if constraint_type != "eq":
            assert init_value > 0, "Inequality constraints must have non-negative initial multiplier values"

            # Adjust initial value based on parameterization
            if parameterization == "softplus":
                init_value = torch.log(torch.exp(torch.tensor(init_value)) - 1)
            elif parameterization == "exp":
                init_value = torch.log(torch.tensor(init_value))
            elif parameterization == "none":
                pass
            else:
                raise ValueError(f"Invalid multiplier parameterization {parameterization}")
        else:
            assert parameterization is None, "Equality constraints must have no parameterization"

        # Initialize Lagrange multiplier as trainable parameter
        self.lagrange = nn.Parameter(torch.full(constraint_shape, init_value, dtype=torch.float32))

    def forward(self, lhs: Optional[torch.Tensor] = None, rhs: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get base multiplier value from parameter
        multiplier = self.lagrange

        # Apply parameterization for inequality constraints
        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = torch.nn.functional.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = torch.exp(multiplier)
            elif self.parameterization == "none":
                pass
            else:
                raise ValueError(f"Invalid multiplier parameterization {self.parameterization}")

        # Return raw multiplier if no constraint values provided
        if lhs is None:
            return multiplier

        # Set rhs to zero tensor if not provided
        if rhs is None:
            rhs = torch.zeros_like(lhs)

        # Calculate difference between left and right hand sides of constraint
        diff = lhs - rhs

        # Ensure shape consistency between multiplier and constraint difference
        assert diff.shape == multiplier.shape, f"Shape mismatch: {diff.shape} vs {multiplier.shape}"

        # Compute Lagrange penalty based on constraint type
        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff
        else:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")


# Preconfigured Lagrange multiplier for greater-or-equal constraints (softplus parameterization)
GeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="geq", parameterization="softplus")

# Preconfigured Lagrange multiplier for less-or-equal constraints (softplus parameterization)
LeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="leq", parameterization="softplus")

# Preconfigured Lagrange multiplier for less-or-equal constraints (no parameterization)
BetterLeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="leq", parameterization="none")

from typing import Any, Mapping, MutableMapping
import gymnasium


class UnnormalizeActionProprio(gymnasium.ActionWrapper, gymnasium.ObservationWrapper):
    """Wrapper to invert normalization for actions and obs['proprio'].

    Uses dataset stats to reverse z-score/ min-max scaling for actions and proprio observations.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        action_proprio_metadata: Mapping[str, Mapping[str, Any]],
        normalization_type: str = "normal",
    ) -> None:
        super().__init__(env)
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type

    def unnormalize(self, data: Any, metadata: Mapping[str, Any]) -> Any:
        """Invert normalization using metadata (normal/bounds scaling).

        Args:
            data: Normalized data to unnormalize.
            metadata: Stats for inversion (mean/std or min/max).

        Returns:
            Unnormalized data in original scale.

        Raises:
            ValueError: If normalization_type is invalid.
        """
        if self.normalization_type == "normal":
            return (data * metadata["std"]) + metadata["mean"]
        if self.normalization_type == "bounds":
            return (data * (metadata["max"] - metadata["min"])) + metadata["min"]
        raise ValueError(f"Unknown normalization_type: {self.normalization_type!r}")

    def action(self, action: Any) -> Any:
        """Unnormalize action using action metadata.

        Args:
            action: Normalized action to unnormalize.

        Returns:
            Unnormalized action.
        """
        return self.unnormalize(action, self.action_proprio_metadata["action"])

    def observation(self, obs: Any) -> Any:
        """Unnormalize obs['proprio'] (if present) using proprio metadata.

        Args:
            obs: Observation to process (dict or other type).

        Returns:
            Observation with unnormalized proprio (if applicable).
        """
        if isinstance(obs, MutableMapping) and "proprio" in obs:
            out = dict(obs)
            out["proprio"] = self.unnormalize(out["proprio"], self.action_proprio_metadata["proprio"])
            return out
        return obs

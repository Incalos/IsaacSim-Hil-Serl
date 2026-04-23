from typing import List, Dict, Any, Optional
import gymnasium
from gymnasium.spaces import flatten_space, flatten, Dict as GymDict


class SERLObsWrapper(gymnasium.ObservationWrapper):
    """Wrapper to flatten proprioceptive state and retain image observations.

    Restructures obs space to include flattened 'state' + original 'images'.
    """

    def __init__(self, env: gymnasium.Env, proprio_keys: Optional[List[str]] = None):
        super().__init__(env)
        self.proprio_keys = proprio_keys or list(self.env.observation_space["state"].keys())
        self._setup_proprio_space()
        self._setup_observation_space()

    def _setup_proprio_space(self) -> None:
        """Initialize proprioceptive space from specified state keys."""
        self.proprio_space = GymDict({key: self.env.observation_space["state"][key] for key in self.proprio_keys})

    def _setup_observation_space(self) -> None:
        """Define new obs space with flattened state and original images."""
        self.observation_space = GymDict(
            {
                "state": flatten_space(self.proprio_space),
                **self.env.observation_space["images"],
            }
        )

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten proprio state and combine with image observations.

        Args:
            obs: Original obs dict with 'state' and 'images' keys.

        Returns:
            Restructured obs with flattened 'state' and original 'images'.
        """
        return {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **obs["images"],
        }

    def reset(self, **kwargs) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset env and process initial observation.

        Args:
            **kwargs: Additional args for underlying env reset.

        Returns:
            (processed_obs, info) from env.
        """
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def flatten_observations(obs: Dict[str, Any], proprio_space: GymDict, proprio_keys: List[str]) -> Dict[str, Any]:
    """Flatten proprio state and retain image data in observation.

    Args:
        obs: Original obs dict with 'state' and 'images' keys.
        proprio_space: Gym Dict space for proprio state.
        proprio_keys: Keys to extract from original state.

    Returns:
        Obs with flattened 'state' and original 'images'.
    """
    return {
        "state": flatten(
            proprio_space,
            {key: obs["state"][key] for key in proprio_keys},
        ),
        **obs["images"],
    }

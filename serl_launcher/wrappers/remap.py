from typing import Any, Dict
import gymnasium


def _tree_map_observation(fn: Any, structure: Any) -> Any:
    """Recursively apply function to leaves of nested tuple/dict structure.

    Args:
        fn: Function to apply to each leaf element.
        structure: Nested structure (tuple/dict/scalar) to process.

    Returns:
        New structure with function applied to all leaves.
    """
    if isinstance(structure, tuple):
        return tuple(_tree_map_observation(fn, v) for v in structure)
    elif isinstance(structure, dict):
        return {k: _tree_map_observation(fn, v) for k, v in structure.items()}
    else:
        return fn(structure)


class RemapWrapper(gymnasium.ObservationWrapper):
    """Wrapper to remap observations to a new structure (tuple/dict/string).

    Restructures observation space and observations per provided new_structure.
    """

    def __init__(self, env: gymnasium.Env, new_structure: Any):
        super().__init__(env)
        self.new_structure = new_structure
        self._setup_observation_space()

    def _setup_observation_space(self) -> None:
        """Initialize observation space based on new_structure type."""
        if isinstance(self.new_structure, tuple):
            self.observation_space = gymnasium.spaces.Tuple([self.env.observation_space[v] for v in self.new_structure])
        elif isinstance(self.new_structure, dict):
            self.observation_space = gymnasium.spaces.Dict(
                {k: self.env.observation_space[v] for k, v in self.new_structure.items()}
            )
        elif isinstance(self.new_structure, str):
            self.observation_space = self.env.observation_space[self.new_structure]
        else:
            raise TypeError(f"Unsupported new_structure type: {type(self.new_structure)}")

    def observation(self, observation: Dict[str, Any]) -> Any:
        """Remap observation to new structure using leaf lookup.

        Args:
            observation: Original observation dict from env.

        Returns:
            Observation restructured to new_structure.
        """
        return _tree_map_observation(lambda x: observation[x], self.new_structure)

from typing import Any
import gymnasium as gym


def _tree_map_observation(fn, structure):
    # Recursively apply fn to all leaves in a nested (tuple/dict/leaf) structure
    if isinstance(structure, tuple):
        return tuple(_tree_map_observation(fn, v) for v in structure)
    elif isinstance(structure, dict):
        return {k: _tree_map_observation(fn, v) for k, v in structure.items()}
    else:
        return fn(structure)


class RemapWrapper(gym.ObservationWrapper):
    # Observation wrapper that reshapes a dict observation into a new key-based structure
    def __init__(self, env: gym.Env, new_structure: Any):
        # new_structure defines how to index into the original dict observation
        super().__init__(env)
        self.new_structure = new_structure
        # Build a matching observation space for the remapped structure
        if isinstance(new_structure, tuple):
            self.observation_space = gym.spaces.Tuple([env.observation_space[v] for v in new_structure])
        elif isinstance(new_structure, dict):
            self.observation_space = gym.spaces.Dict({k: env.observation_space[v] for k, v in new_structure.items()})
        elif isinstance(new_structure, str):
            self.observation_space = env.observation_space[new_structure]
        else:
            raise TypeError(f"Unsupported type {type(new_structure)}")

    def observation(self, observation):
        # Remap incoming dict observation to the specified nested structure
        return _tree_map_observation(lambda x: observation[x], self.new_structure)

from typing import Any
import gymnasium as gym


def _tree_map_observation(fn, structure):
    # Recursively apply function to all leaves in nested tuple/dict/leaf structure
    if isinstance(structure, tuple):
        return tuple(_tree_map_observation(fn, v) for v in structure)
    elif isinstance(structure, dict):
        return {k: _tree_map_observation(fn, v) for k, v in structure.items()}
    else:
        return fn(structure)


class RemapWrapper(gym.ObservationWrapper):
    # Observation wrapper to reshape dict observation into new key-based structure
    def __init__(self, env: gym.Env, new_structure: Any):
        # new_structure defines indexing rules for original dict observation
        super().__init__(env)
        self.new_structure = new_structure

        # Build observation space matching the remapped structure
        if isinstance(new_structure, tuple):
            self.observation_space = gym.spaces.Tuple([env.observation_space[v] for v in new_structure])
        elif isinstance(new_structure, dict):
            self.observation_space = gym.spaces.Dict({k: env.observation_space[v] for k, v in new_structure.items()})
        elif isinstance(new_structure, str):
            self.observation_space = env.observation_space[new_structure]
        else:
            raise TypeError(f"Unsupported type {type(new_structure)}")

    def observation(self, observation):
        # Remap input dict observation to specified nested structure
        return _tree_map_observation(lambda x: observation[x], self.new_structure)

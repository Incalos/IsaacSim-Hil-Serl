import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    # Observation wrapper that flattens selected state (proprio) and keeps images separate
    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        # Select which keys from the state dict are treated as proprioceptive features
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())
        # Build a sub-dict space containing only the chosen proprio keys
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )
        # Observation space consists of a flattened state vector plus image observations
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        # Flatten selected proprio keys and merge with image observations
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        # Reset env and immediately convert observation to flattened format
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def flatten_observations(obs, proprio_space, proprio_keys):
    # Utility function to apply the same flattening logic outside of the wrapper
    obs = {
        "state": flatten(
            proprio_space,
            {key: obs["state"][key] for key in proprio_keys},
        ),
        **(obs["images"]),
    }
    return obs

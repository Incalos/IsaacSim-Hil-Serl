import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        # Set proprioceptive keys (default to all state keys if None)
        self.proprio_keys = proprio_keys if proprio_keys is not None else list(self.env.observation_space["state"].keys())

        # Create subspace for selected proprioceptive features
        self.proprio_space = gym.spaces.Dict({key: self.env.observation_space["state"][key] for key in self.proprio_keys})

        # Define new observation space: flattened proprio state + original image spaces
        self.observation_space = gym.spaces.Dict({
            "state": flatten_space(self.proprio_space),
            **(self.env.observation_space["images"]),
        })

    def observation(self, obs):
        # Flatten proprioceptive state and merge with image observations
        flattened_obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
        }
        return flattened_obs

    def reset(self, **kwargs):
        # Reset environment and process observation
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def flatten_observations(obs, proprio_space, proprio_keys):
    # Utility: apply same flattening logic as the wrapper
    flattened_obs = {
        "state": flatten(
            proprio_space,
            {key: obs["state"][key] for key in proprio_keys},
        ),
        **(obs["images"]),
    }
    return flattened_obs

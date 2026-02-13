import gymnasium as gym


class UnnormalizeActionProprio(gym.ActionWrapper, gym.ObservationWrapper):
    # Wrapper that converts normalized actions and proprioception back to original scale
    def __init__(
        self,
        env: gym.Env,
        action_proprio_metadata: dict,
        normalization_type: str = "normal",
    ):
        # Store metadata and chosen normalization scheme
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        super().__init__(env)

    def unnormalize(self, data, metadata):
        # Reverse normalization using either mean/std or min/max bounds
        if self.normalization_type == "normal":
            return (data * metadata["std"]) + metadata["mean"]
        elif self.normalization_type == "bounds":
            return (data * (metadata["max"] - metadata["min"])) + metadata["min"]
        else:
            raise ValueError(f"Unknown action/proprio normalization type: {self.normalization_type}")

    def action(self, action):
        # Unnormalize action before passing to the wrapped environment
        return self.unnormalize(action, self.action_proprio_metadata["action"])

    def observation(self, obs):
        # Unnormalize proprioceptive part of the observation
        obs["proprio"] = self.unnormalize(obs["proprio"], self.action_proprio_metadata["proprio"])
        return obs

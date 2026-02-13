import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from .devices.so101_gamepad import GamepadExpert
from scipy.spatial.transform import Rotation as R
from typing import List

# Standard logistic function used to squash classifier logits into a [0, 1] probability range:
# $sigmoid(x) = \frac{1}{1 + e^{-x}}$
sigmoid = lambda x: 1 / (1 + np.exp(-x))


class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        # Blocks execution at the end of an episode to allow a human to manually label success/failure
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info["succeed"] = rew
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        # Binary reward often serves as a terminal signal in sparse-reward environments
        done = done or rew
        info["succeed"] = bool(rew)
        # Enforce consistent control frequency if a target Hz is specified
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)

    def compute_reward(self, obs):
        # Accumulates rewards based on a sequence of classifiers for multi-part tasks
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue
            logit = classifier_func(obs).item()
            # Threshold set to 0.75 for robust stage-completion detection
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1
        return sum(rewards)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or all(self.received)
        info["succeed"] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info["succeed"] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        # Re-defines the observation space to accommodate 6D pose (XYZ + Euler) instead of 7D (XYZ + Quat)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(-np.inf, np.inf, shape=(6,))

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        # Converts pose orientation to XYZ intrinsic Euler angles for simplified feature learning
        observation["state"]["tcp_pose"] = np.concatenate((tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz")))
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # Map 7D Quat pose to a 9D representation (3D Position + 6D continuous rotation representation)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(-np.inf, np.inf, shape=(9,))

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        # Uses the first two columns of the rotation matrix (6 elements) to avoid quaternion discontinuities
        observation["state"]["tcp_pose"] = np.concatenate((tcp_pose[:3], r[..., :2].flatten()))
        return observation


class GamepadIntervention(gym.Wrapper):
    def __init__(self, env, guid):
        super().__init__(env)
        self.expert = GamepadExpert(guid=guid)

    def action(self, action):
        deltas, intervened = self.expert.get_action()
        if intervened:
            d_sh, d_y, d_z, d_flex, d_roll, d_gr = deltas
            final_action = np.array([d_sh, d_y, d_z, d_flex, d_roll, d_gr])
            return final_action, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action.copy()
        return obs, rew, done, truncated, info

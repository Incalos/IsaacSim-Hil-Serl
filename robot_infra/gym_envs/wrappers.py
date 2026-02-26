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
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue
            if classifier_func(obs):
                self.received[i] = True
        return sum(self.received)

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
        self.env = env.unwrapped
        self.g_min = env.unwrapped.gripper_limits[0]
        self.g_distance = env.unwrapped.gripper_limits[1] - self.g_min
        self.scale = (0.05, 0.01, 0.25)
        self.action_scale = env.unwrapped.action_scale

    def calculate_delta_eef(self, curr_eef, next_eef):
        p_curr = np.array(curr_eef[:3])
        q_curr = R.from_quat(curr_eef[3:7])
        p_next = np.array(next_eef[:3])
        q_next = R.from_quat(next_eef[3:7])
        delta_p = p_next - p_curr
        delta_r_obj = q_next * q_curr.inv()
        delta_axis_angle = delta_r_obj.as_rotvec()
        delta_eef = np.concatenate([delta_p / self.action_scale[0], delta_axis_angle / self.action_scale[1]])
        return delta_eef

    def action(self, action):
        deltas, intervened = self.expert.get_action()
        if intervened:
            d_sh, d_y, d_z, d_flex, d_roll, d_gr = deltas
            curr_eef = self.env.curr_eef_poses_quat.copy()
            next_joint_positions = self.env.curr_joint_positions.copy()
            next_joint_positions[0] += d_sh * self.scale[0]
            next_joint_positions[3] += d_flex * self.scale[0]
            next_joint_positions[4] += d_roll * self.scale[0]
            gripper_action = 2 * (next_joint_positions[-1] + d_gr * self.scale[2] - self.g_min) / self.g_distance - 1
            nextpos = self.env.joints_to_eef_fk(next_joint_positions)
            nextpos[1] += d_y * self.scale[1]
            nextpos[2] += d_z * self.scale[1]
            res = self.calculate_delta_eef(curr_eef, nextpos)
            final_action = np.concatenate([res, [gripper_action / self.action_scale[2]]])
            return final_action, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action.copy()
        return obs, rew, done, truncated, info

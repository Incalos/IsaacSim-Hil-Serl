from gymnasium import Env
import gymnasium as gym
import numpy as np
from .devices.so101_gamepad import GamepadExpert
from scipy.spatial.transform import Rotation as R
from typing import List
import time


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):

    def __init__(self, env: Env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    # Calculate reward based on observation and action
    def compute_reward(self, obs, action):
        total_reward = -0.1
        action_norm = np.linalg.norm(action[:6])
        total_reward -= 0.05 * action_norm
        r, success = self.reward_classifier_func(obs)
        total_reward += r
        return total_reward, success

    # Override step method to integrate custom reward calculation and frequency control
    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew, success = self.compute_reward(obs, action)
        done = done or success
        info["succeed"] = success

        # Enforce consistent control frequency if target Hz is specified
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    # Reset environment and reset success flag
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):

    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)

    # Calculate multi-stage reward based on sequential classifiers
    def compute_reward(self, obs, action):
        total_reward = -0.1
        action_norm = np.linalg.norm(action[:6])
        total_reward -= 0.05 * action_norm

        # Process each stage in order until an unfinished stage is found
        for i in range(len(self.reward_classifier_func)):
            if not self.received[i]:
                rew, success = self.reward_classifier_func[i](obs)
                total_reward += rew
                if success:
                    self.received[i] = True
                break
        return total_reward

    # Override step method for multi-stage reward calculation
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs, action)
        done = done or all(self.received)
        info["succeed"] = all(self.received)
        return obs, rew, done, truncated, info

    # Reset environment and multi-stage progress tracking
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info["succeed"] = False
        return obs, info


class GamepadIntervention(gym.Wrapper):

    def __init__(self, env, guid):
        super().__init__(env)
        self.expert = GamepadExpert(guid=guid)
        self.env = env.unwrapped
        self.g_min = env.unwrapped.gripper_limits[0]
        self.g_distance = env.unwrapped.gripper_limits[1] - self.g_min
        self.scale = (0.05, 0.01, 0.25)
        self.action_scale = env.unwrapped.action_scale

    # Calculate end-effector delta (position + rotation) between current and next state
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

    # Process gamepad intervention to override agent action if needed
    def action(self, action):
        deltas, intervened = self.expert.get_action()

        # Replace action with gamepad input if intervention is triggered
        if intervened:
            d_sh, d_y, d_z, d_flex, d_roll, d_gr = deltas
            curr_eef = self.env.curr_eef_poses_quat.copy()
            next_joint_positions = self.env.curr_joint_positions.copy()

            # Update joint positions based on gamepad input
            next_joint_positions[0] += d_sh * self.scale[0]
            next_joint_positions[3] += d_flex * self.scale[0]
            next_joint_positions[4] += d_roll * self.scale[0]

            # Calculate normalized gripper action
            gripper_action = 2 * (next_joint_positions[-1] + d_gr * self.scale[2] - self.g_min) / self.g_distance - 1

            # Forward kinematics to get new end-effector position
            nextpos = self.env.joints_to_eef_fk(next_joint_positions)
            nextpos[1] += d_y * self.scale[1]
            nextpos[2] += d_z * self.scale[1]

            # Calculate final action from end-effector delta
            res = self.calculate_delta_eef(curr_eef, nextpos)
            final_action = np.concatenate([res, [gripper_action / self.action_scale[2]]])
            return final_action, True

        return action, False

    # Override step method to integrate gamepad intervention
    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        # Record intervened action in info if replacement occurred
        if replaced:
            info["intervene_action"] = new_action.copy()
        return obs, rew, done, truncated, info

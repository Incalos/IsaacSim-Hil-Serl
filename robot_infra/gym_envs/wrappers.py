from gymnasium import Env
import gymnasium as gym
import numpy as np
from .devices.so101_gamepad import GamepadExpert
from typing import List, Tuple, Dict, Any, Sequence
import time
from pathlib import Path
from pytransform3d.urdf import UrdfTransformManager
from scipy.spatial.transform import Rotation


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):

    def __init__(self, env: Env, reward_classifier_func, target_hz=10):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs, action):
        # Calculate action norm penalty
        action_norm = np.linalg.norm(action[:6])
        total_reward = -0.05 * action_norm

        # Get classifier reward and success flag
        r, success = self.reward_classifier_func(obs)
        total_reward += r

        return total_reward, success

    def step(self, action):
        # Record start time for frequency control
        start_time = time.time()

        # Execute original environment step
        obs, rew, done, truncated, info = self.env.step(action)

        # Compute custom reward and update success state
        rew, success = self.compute_reward(obs, action)
        done = done or success
        info["succeed"] = success

        # Enforce target control frequency
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        # Reset environment and success flag
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False

        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):

    def __init__(self, env: Env, reward_classifier_func: List[callable], target_hz=10):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
        self.target_hz = target_hz

    def compute_reward(self, obs, action):
        # Calculate action norm penalty
        action_norm = np.linalg.norm(action[:6])
        total_reward = -0.05 * action_norm

        # Process each stage sequentially until unfinished stage found
        for i in range(len(self.reward_classifier_func)):
            if not self.received[i]:
                rew, success = self.reward_classifier_func[i](obs)
                total_reward += rew
                if success:
                    self.received[i] = True
                break

        return total_reward

    def step(self, action):
        # Record start time for frequency control
        start_time = time.time()

        # Execute original environment step
        obs, _, done, truncated, info = self.env.step(action)

        # Compute multi-stage reward
        rew = self.compute_reward(obs, action)

        # Check if all stages completed
        all_done = all(self.received)
        done = done or all_done

        # Update info with stage progress and custom reward
        info.update({
            "succeed": all_done,
            "stage_completed": self.received.copy(),
            "custom_reward": rew,
        })

        # Enforce target control frequency
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        # Reset environment and stage progress tracking
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)

        # Update info with reset state
        info.update({
            "succeed": False,
            "stage_completed": self.received.copy(),
        })

        return obs, info


class GamepadIntervention(gym.Wrapper):

    def __init__(self, env: Env, guid: str):
        super().__init__(env)
        self.expert = GamepadExpert(guid=guid)
        self.env_unwrapped = env.unwrapped

        # Initialize gripper parameters
        self.gripper_min = self.env_unwrapped.gripper_limits[0]
        self.gripper_range = self.env_unwrapped.gripper_limits[1] - self.gripper_min

        # Initialize scale factors
        self.scale = (0.05, 0.01, 0.25)
        self.action_scale = self.env_unwrapped.action_scale

        self.urdf_path = self.env_unwrapped.urdf_path
        self.joint_names = self.env_unwrapped.joint_names

        self._load_urdf()

    def _load_urdf(self):
        # Resolve URDF file path
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            _root = Path(__file__).resolve().parent.parent
            urdf_path = _root / urdf_path

        # Load URDF for forward kinematics
        self.tm = UrdfTransformManager()
        with open(urdf_path, "r") as f:
            urdf_text = f.read()
        self.tm.load_urdf(urdf_text)

    def joints_to_eef_fk(self, joint_positions: Sequence[float]) -> np.ndarray:
        # Set joint positions in transform manager
        for name, pos in zip(self.joint_names, joint_positions[:-1]):
            self.tm.set_joint(name, pos)

        # Get end effector transform
        T = self.tm.get_transform("so101_new_calib_gripper", "so101_new_calib")
        position = T[:3, 3]
        quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()

        # Ensure quaternion w component is positive for consistency
        if quat_xyzw[3] < 0:
            quat_xyzw = -quat_xyzw

        return np.concatenate([position, quat_xyzw])

    def calculate_delta_eef(self, curr_eef: np.ndarray, next_eef: np.ndarray) -> np.ndarray:
        # Extract position components
        p_curr = np.array(curr_eef[:3])
        p_next = np.array(next_eef[:3])
        delta_pos = p_next - p_curr

        # Calculate rotation delta (axis-angle representation)
        q_curr = Rotation.from_quat(curr_eef[3:7])
        q_next = Rotation.from_quat(next_eef[3:7])
        delta_rot = q_next * q_curr.inv()
        delta_rot_axisangle = delta_rot.as_rotvec()

        # Normalize position and rotation deltas
        delta_pos_norm = delta_pos / self.action_scale[0]
        delta_rot_norm = delta_rot_axisangle / self.action_scale[1]
        delta_eef = np.concatenate([delta_pos_norm, delta_rot_norm])

        return delta_eef

    def process_gamepad_action(self, agent_action: np.ndarray) -> Tuple[np.ndarray, bool]:
        # Get gamepad input and intervention flag
        deltas, intervened = self.expert.get_action()

        # Return original agent action if no intervention
        if not intervened:
            return agent_action, False

        # Parse gamepad delta values
        d_sh, d_y, d_z, d_flex, d_roll, d_gr = deltas

        # Get current robot state
        curr_eef = self.env_unwrapped.curr_eef_poses_quat.copy()
        next_joint_pos = self.env_unwrapped.curr_joint_positions.copy()

        # Update joint positions based on gamepad input
        next_joint_pos[0] += d_sh * self.scale[0]
        next_joint_pos[3] += d_flex * self.scale[0]
        next_joint_pos[4] += d_roll * self.scale[0]

        # Calculate normalized gripper action
        gripper_target = next_joint_pos[-1] + d_gr * self.scale[2]
        gripper_action = 2 * (gripper_target - self.gripper_min) / self.gripper_range - 1
        gripper_action = np.clip(gripper_action, -1.0, 1.0)

        # Calculate new end effector pose using forward kinematics
        next_eef = self.joints_to_eef_fk(next_joint_pos)
        next_eef[1] += d_y * self.scale[1]
        next_eef[2] += d_z * self.scale[1]

        # Calculate final action from end effector delta
        delta_eef = self.calculate_delta_eef(curr_eef, next_eef)
        final_action = np.concatenate([delta_eef, [gripper_action / self.action_scale[2]]])

        return final_action, True

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Process gamepad intervention (override agent action if needed)
        new_action, intervened = self.process_gamepad_action(action)
        # Execute environment step with modified action
        obs, rew, done, truncated, info = self.env_unwrapped.step(new_action)

        # Record intervention action in info if applicable
        if intervened:
            info["intervene_action"] = new_action.copy()

        return obs, rew, done, truncated, info

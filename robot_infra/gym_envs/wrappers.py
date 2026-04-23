import gymnasium
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Sequence, Callable
from pytransform3d.urdf import UrdfTransformManager
from scipy.spatial.transform import Rotation
from .devices.so101_gamepad import GamepadExpert


class MultiCameraBinaryRewardClassifierWrapper(gymnasium.Wrapper):
    """Wrapper for binary reward calculation using camera-based classifier.

    Args:
        env: Base gym environment to wrap
        reward_classifier_func: Function to compute binary reward from observations
        target_hz: Target control frequency (default: 10)
    """

    def __init__(
        self, env: gymnasium.Env, reward_classifier_func: Callable[[Dict[str, Any]], float], target_hz: int = 10
    ):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs: Dict[str, Any]) -> float:
        """Compute binary reward using classifier function.

        Args:
            obs: Environment observation

        Returns:
            Binary reward value (0 or positive value)
        """
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0.0

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step environment with reward calculation and frequency control.

        Args:
            action: Action array for environment step

        Returns:
            Tuple containing observation, computed reward, done flag, truncated flag, info dict
        """
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or bool(rew)
        info["succeed"] = bool(rew)

        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and initialize success flag.

        Args:
            **kwargs: Additional reset arguments for base environment

        Returns:
            Tuple containing initial observation and info dict
        """
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gymnasium.Wrapper):
    """Wrapper for multi-stage binary reward calculation.

    Args:
        env: Base gym environment to wrap
        reward_classifier_func: List of classifier functions for each reward stage
        target_hz: Target control frequency (default: 10)
    """

    def __init__(
        self, env: gymnasium.Env, reward_classifier_func: List[Callable[[Dict[str, Any]], float]], target_hz: int = 10
    ):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
        self.target_hz = target_hz

    def compute_reward(self, obs: Dict[str, Any]) -> float:
        """Compute reward by processing stages in order until first uncompleted stage.

        Args:
            obs: Environment observation

        Returns:
            Cumulative reward from completed stages
        """
        total_reward = 0.0
        for i in range(len(self.reward_classifier_func)):
            if not self.received[i]:
                rew = self.reward_classifier_func[i](obs)
                total_reward += rew
                if bool(rew):
                    self.received[i] = True
                break
        return total_reward

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step environment with multi-stage reward calculation and frequency control.

        Args:
            action: Action array for environment step

        Returns:
            Tuple containing observation, computed reward, done flag, truncated flag, info dict
        """
        start_time = time.time()
        obs, _, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        all_done = all(self.received)
        done = done or all_done

        info.update(
            {
                "succeed": all_done,
                "stage_completed": self.received.copy(),
                "custom_reward": rew,
            }
        )

        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and stage tracking flags.

        Args:
            **kwargs: Additional reset arguments for base environment

        Returns:
            Tuple containing initial observation and info dict
        """
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)

        info.update(
            {
                "succeed": False,
                "stage_completed": self.received.copy(),
            }
        )
        return obs, info


class Quat2EulerWrapper(gymnasium.ObservationWrapper):
    """Wrapper to convert TCP pose quaternion to Euler angles (xyz).

    Converts tcp_pose from [x,y,z,qx,qy,qz,qw] to [x,y,z,rx,ry,rz]
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = gymnasium.spaces.Box(-np.inf, np.inf, shape=(6,))

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Transform TCP pose quaternion to Euler angles.

        Args:
            observation: Original observation with quaternion-based tcp_pose

        Returns:
            Observation with Euler angle-based tcp_pose
        """
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], Rotation.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gymnasium.ObservationWrapper):
    """Wrapper to convert TCP pose quaternion to rotation matrix columns.

    Converts tcp_pose from [x,y,z,qx,qy,qz,qw] to [x,y,z, R[:,0], R[:,1]] (flattened)
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = gymnasium.spaces.Box(-np.inf, np.inf, shape=(9,))

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Transform TCP pose quaternion to rotation matrix columns.

        Args:
            observation: Original observation with quaternion-based tcp_pose

        Returns:
            Observation with rotation matrix column-based tcp_pose
        """
        tcp_pose = observation["state"]["tcp_pose"]
        r = Rotation.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate((tcp_pose[:3], r[..., :2].flatten()))
        return observation


class GamepadIntervention(gymnasium.Wrapper):
    """Wrapper for gamepad-based manual intervention in robot control.

    Args:
        env: Base gym environment to wrap
        robot_params: Dictionary with robot parameters
        robot_urdf_path: Path to robot URDF file
    """

    def __init__(self, env: gymnasium.Env, robot_params: dict, robot_urdf_path: str):
        super().__init__(env)
        self.expert = GamepadExpert(guid=robot_params["gamepad_guid"])
        self.env_unwrapped = env.unwrapped
        self.scale = (0.05, 0.01, 0.4)
        self.action_scale = robot_params["action_scale"]
        self.joint_names = robot_params["joint_names"]
        self._load_urdf(urdf_path=robot_urdf_path)

    def _load_urdf(self, urdf_path: str) -> None:
        """Load robot URDF for forward kinematics calculations.

        Args:
            urdf_path: Path to robot URDF file
        """
        self.tm = UrdfTransformManager()
        with open(urdf_path, "r") as f:
            urdf_text = f.read()
        self.tm.load_urdf(urdf_text)

    def joints_to_eef_fk(self, joint_positions: Sequence[float]) -> np.ndarray:
        """Compute end-effector pose from joint positions using forward kinematics.

        Args:
            joint_positions: Sequence of joint positions (last element ignored)

        Returns:
            End-effector pose [x,y,z,qx,qy,qz,qw] (quaternion with positive w)
        """
        for name, pos in zip(self.joint_names, joint_positions[:-1]):
            self.tm.set_joint(name, pos)
        T = self.tm.get_transform("gripper_link", "base_link")
        position = T[:3, 3]
        quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()

        if quat_xyzw[3] < 0:
            quat_xyzw = -quat_xyzw
        return np.concatenate([position, quat_xyzw])

    def calculate_delta_eef(self, curr_eef: np.ndarray, next_eef: np.ndarray) -> np.ndarray:
        """Calculate normalized delta between current and next end-effector pose.

        Args:
            curr_eef: Current end-effector pose [x,y,z,qx,qy,qz,qw]
            next_eef: Target end-effector pose [x,y,z,qx,qy,qz,qw]

        Returns:
            Normalized delta pose [dx,dy,dz,rx,ry,rz] (axis-angle rotation)
        """
        p_curr = np.array(curr_eef[:3])
        p_next = np.array(next_eef[:3])
        delta_pos = p_next - p_curr

        q_curr = Rotation.from_quat(curr_eef[3:7])
        q_next = Rotation.from_quat(next_eef[3:7])
        delta_rot = q_next * q_curr.inv()
        delta_rot_axisangle = delta_rot.as_rotvec()

        delta_pos_norm = delta_pos / self.action_scale[0]
        delta_rot_norm = delta_rot_axisangle / self.action_scale[1]
        delta_eef = np.concatenate([delta_pos_norm, delta_rot_norm])
        return delta_eef

    def process_gamepad_action(self, agent_action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process gamepad input to generate intervention action.

        Args:
            agent_action: Original agent action (retained if no intervention)

        Returns:
            Tuple containing modified action and intervention flag
        """
        deltas, intervened = self.expert.get_action()
        if not intervened:
            return agent_action, False

        d_sh, d_forward, d_z, d_flex, d_roll, d_gr = deltas
        curr_eef = self.env_unwrapped.curr_eef_poses_quat.copy()
        next_joint_pos = self.env_unwrapped.curr_joint_positions.copy()

        next_joint_pos[0] += d_sh * self.scale[0]
        next_joint_pos[3] += d_flex * self.scale[0]
        next_joint_pos[4] += d_roll * self.scale[0]

        next_eef = self.joints_to_eef_fk(next_joint_pos)
        next_eef[0] += d_forward * self.scale[1] * float(np.cos(next_joint_pos[0]))
        next_eef[1] += d_forward * self.scale[1] * float(np.sin(-1 * next_joint_pos[0]))
        next_eef[2] += d_z * self.scale[1]

        delta_eef = self.calculate_delta_eef(curr_eef, next_eef)
        final_action = np.concatenate([delta_eef, [d_gr * self.scale[2] / self.action_scale[2]]])
        return final_action, True

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step environment with gamepad intervention if active.

        Args:
            action: Original agent action

        Returns:
            Tuple containing observation, reward, done flag, truncated flag, info dict
        """
        new_action, intervened = self.process_gamepad_action(action)
        obs, rew, done, truncated, info = self.env_unwrapped.step(new_action)

        if intervened:
            info["intervene_action"] = new_action.copy()
        return obs, rew, done, truncated, info

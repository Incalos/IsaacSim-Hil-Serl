import copy
import gymnasium
import numpy as np
from scipy.spatial.transform import Rotation as R
from gym import Env
from typing import Tuple, Dict, Any
from .transformations import construct_transform_matrix, construct_homogeneous_matrix


class RelativeFrame(gymnasium.Wrapper):
    """Wrapper to transform observations/actions between end-effector and base frames.

    Args:
        env: Base gym environment to wrap
        include_relative_pose: Whether to include relative pose in observations (default: True)
    """

    def __init__(self, env: Env, include_relative_pose: bool = True):
        super().__init__(env)
        self.transform_matrix = np.zeros((6, 6))
        self.include_relative_pose = include_relative_pose

        if self.include_relative_pose:
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step environment with action transformed from end-effector to base frame.

        Args:
            action: Action array (first 6 DOF for end-effector control)

        Returns:
            Tuple containing transformed observation, reward, done flag, truncated flag, info dict
        """
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and initialize frame transformation matrices.

        Args:
            **kwargs: Additional reset arguments for base environment

        Returns:
            Tuple containing transformed initial observation and info dict
        """
        obs, info = self.env.reset(**kwargs)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])

        if self.include_relative_pose:
            self.T_r_o_inv = np.linalg.inv(construct_homogeneous_matrix(obs["state"]["tcp_pose"]))

        return self.transform_observation(obs), info

    def transform_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform observation from base frame to end-effector frame.

        Args:
            obs: Original observation from base environment

        Returns:
            Transformed observation with end-effector frame coordinates
        """
        transform_inv = np.linalg.inv(self.transform_matrix)
        obs["state"]["tcp_vel"] = transform_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """Transform action from end-effector frame to base frame.

        Args:
            action: Action in end-effector frame (first 6 DOF)

        Returns:
            Action transformed to base frame
        """
        action = np.array(action)
        action[:6] = self.transform_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray) -> np.ndarray:
        """Inverse transform action from base frame to end-effector frame.

        Args:
            action: Action in base frame (first 6 DOF)

        Returns:
            Action transformed to end-effector frame
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.transform_matrix) @ action[:6]
        return action

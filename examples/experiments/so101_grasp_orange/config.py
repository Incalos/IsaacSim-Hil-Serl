import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import numpy as np
from pathlib import Path
from robot_infra.gym_envs.wrappers import (
    GamepadIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2EulerWrapper,
)
from robot_infra.gym_envs.so101_env import SO101Env
from robot_infra.gym_envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func


class TrainConfig:
    """Training configuration class for robot environment setup.

    Manages loading experiment parameters, file paths, and environment configuration
    with optional reward classification for grasp detection.
    """

    def __init__(self):
        """Initialize TrainConfig by loading parameters and setting file paths."""
        with open(str(Path(__file__).parent / "exp_params.yaml"), "r") as f:
            self.robot_params = yaml.load(f, Loader=yaml.FullLoader)
        self.demo_path = os.path.join(os.path.dirname(__file__), "demo_data")
        self.classifier_ckpt_path = os.path.join(os.path.dirname(__file__), "classifier_ckpt", "checkpoint.pth")
        self.robot_urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "robot_infra",
            "isaacsim_venvs",
            "tasks",
            "robots",
            self.robot_params["robot_type"],
            "model",
            f"{self.robot_params['robot_type']}.urdf",
        )

    def get_environment(self, fake_env=False, classifier=True) -> SO101Env:
        """Create and configure SO101 environment with optional reward classification.

        Args:
            fake_env: Whether to use fake environment mode (for testing)
            classifier: Whether to enable reward classifier for grasp detection

        Returns:
            Configured SO101 environment instance with applied wrappers
        """
        env = SO101Env(fake_env=fake_env, robot_params=self.robot_params)
        env = GamepadIntervention(env, robot_params=self.robot_params, robot_urdf_path=self.robot_urdf_path)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.robot_params["proprio_keys"])
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier_model = load_classifier_func(
                image_keys=self.robot_params["classifier_keys"],
                checkpoint_path=self.classifier_ckpt_path,
                img_size=self.robot_params["image_size"],
            )

            def reward_func(obs: dict) -> float:

                def sigmoid(x: float) -> float:
                    x = np.clip(x, -10, 10)
                    return 1 / (1 + np.exp(-x))

                logits = classifier_model(obs)
                if np.isnan(logits):
                    logits = -10
                prob = sigmoid(logits)
                gripper_closed = obs["state"][0][5] < 0.6
                is_grasped = (prob > 0.75) and gripper_closed
                return 1.0 if is_grasped else 0.0

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        return env

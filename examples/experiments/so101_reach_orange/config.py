import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from robot_infra.gym_envs.wrappers import GamepadIntervention, MultiCameraBinaryRewardClassifierWrapper
from robot_infra.gym_envs.so101_env import DefaultEnvConfig, SO101Env
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.config import DefaultTrainingConfig
from serl_launcher.networks.reward_classifier import load_classifier_func


# Environment configuration class
class EnvConfig(DefaultEnvConfig):
    ROBOT_CONFIG = "examples/experiments/so101_reach_orange/so101_params.yaml"
    IMAGE_CROP: dict[str, callable] = {
        "front_camera": lambda img: img[50:320, 180:480, :],
        "wrist_camera": lambda img: img,
        "side_camera": lambda img: img[:250, :470, :],
    }
    RANDOM_RESET = False
    ACTION_SCALE = (0.015, 0.1, 1)
    MAX_EPISODE_LENGTH = 100


# Training configuration class
class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_camera", "front_camera", "side_camera"]
    classifier_keys = ["wrist_camera", "front_camera", "side_camera"]
    proprio_keys = ["q", "tcp_pose"]

    buffer_period = 1000
    checkpoint_period = 1000
    steps_per_update = 10000
    fake_env = False
    image_size = (144, 192)
    batch_size = 64
    cta_ratio = 4
    discount = 0.97
    max_steps = 50000
    replay_buffer_capacity = 50000
    random_steps = 0
    training_starts = 1000
    log_period = 100
    eval_period = 1000
    encoder_type = "resnet34-pretrained"
    demo_path = os.path.join(os.path.dirname(__file__), "demo_data")

    # Create and configure environment instance
    def get_environment(self, fake_env=False, save_video=False, classifier=True):
        # Initialize base SO101 environment
        env = SO101Env(fake_env=fake_env, config=EnvConfig(), image_size=self.image_size)
        # Add gamepad intervention capability
        env = GamepadIntervention(env, guid="0300509d5e040000120b000009050000")

        # Apply SERL observation formatting wrapper
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        # Add temporal chunking to observations
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # Add reward classifier if enabled
        if classifier:
            # Load pre-trained reward classification model
            classifier_model = load_classifier_func(
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.join(os.path.dirname(__file__), "classifier_ckpt", "checkpoint.pth"),
                img_size=self.image_size,
            )

            # Sigmoid activation function for probability calculation
            sigmoid = lambda x: 1 / (1 + np.exp(-x))

            # Reward calculation function
            def reward_func(obs):
                prob = sigmoid(classifier_model(obs))
                reward = prob * 2.0
                is_grasped = prob > 0.75 and obs["state"][0][5] < 0.7
                if is_grasped:
                    reward += 5
                return reward, is_grasped

            # Wrap environment with reward classifier
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        return env

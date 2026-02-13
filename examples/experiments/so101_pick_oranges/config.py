import os
import sys
import numpy as np

# Dynamically resolve project root and append to sys.path for local module discovery
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from robot_infra.gym_envs.wrappers import GamepadIntervention, MultiCameraBinaryRewardClassifierWrapper
from robot_infra.gym_envs.so101_env import DefaultEnvConfig, SO101Env
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.config import DefaultTrainingConfig
from serl_launcher.networks.reward_classifier import load_classifier_func


class EnvConfig(DefaultEnvConfig):
    # Networking and hardware-specific camera settings
    SERVER_URL: str = "http://127.0.0.1:5000"
    CAMERA_NAMES: list[str] = ["wrist_camera", "front_camera", "side_camera"]
    # Define ROI cropping for different camera viewpoints to focus on the workspace
    IMAGE_CROP: dict[str, callable] = {
        "front_camera": lambda img: img[100:350, 20:450, :],
        "wrist_camera": lambda img: img,
        "side_camera": lambda img: img[:250, :370, :],
    }
    RANDOM_RESET = False
    ACTION_SCALE = (0.01, 0.05, 0.25)
    MAX_EPISODE_LENGTH = 200


class TrainConfig(DefaultTrainingConfig):
    # Observation space and proprioception configuration
    image_keys = ["wrist_camera", "front_camera", "side_camera"]
    classifier_keys = ["wrist_camera", "front_camera", "side_camera"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "q", "dq"]
    # Training hyperparameters and buffer management
    buffer_period = 1000
    checkpoint_period = 500
    steps_per_update = 50
    fake_env = False
    image_size = (128, 128)
    batch_size = 2048
    cta_ratio = 4
    discount = 0.97
    max_steps = 50000
    replay_buffer_capacity = 200000
    # Warm-up phase before gradient updates begin
    random_steps = 0
    training_starts = 100
    log_period = 100
    eval_period = 100
    encoder_type = "resnet34-pretrained"
    demo_path = os.path.join(os.path.dirname(__file__), "demo_data")

    def get_environment(self, fake_env=False, save_video=False, classifier=True):
        # Initialize base environment and apply hardware/teleop wrappers
        env = SO101Env(fake_env=fake_env, config=EnvConfig())
        env = GamepadIntervention(env, guid="0300509d5e040000120b000009050000")
        # Apply SERL-specific observation formatting and temporal chunking
        # env = RelativeFrame(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            # Load the pre-trained neural network for binary success classification
            classifier = load_classifier_func(
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.join(os.path.dirname(__file__), "classifier_ckpt", "classifier_model.pth"),
                img_size=self.image_size,
            )

            def reward_func(obs):
                # Apply sigmoid activation to the classifier output to get success probability
                # $sigmoid(x) = \frac{1}{1 + e^{-x}}$
                sigmoid = lambda x: 1 / (1 + np.exp(-x))
                # Threshold set at 0.85 for high-confidence reward signals
                return int(sigmoid(classifier(obs)) > 0.8)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env

from abc import abstractmethod
from typing import List


class DefaultTrainingConfig:
    # Core algorithm and trajectory settings
    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 32
    cta_ratio: int = 1
    discount: float = 0.97
    # Training duration and memory management
    max_steps: int = 20000
    replay_buffer_capacity: int = 1000
    # Schedule for initialization and update frequency
    random_steps: int = 0
    training_starts: int = 50
    steps_per_update: int = 100
    # Logging and evaluation intervals
    log_period: int = 100
    eval_period: int = 100
    # Model architecture and demonstration data paths
    encoder_type: str = "resnet50"
    demo_path: str = None
    # Checkpointing and buffer serialization frequency
    checkpoint_period: int = 0
    buffer_period: int = 0
    # Evaluation specific parameters
    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5
    # Input modality definitions for the neural network
    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # Must be implemented to return the specific gym environment instance
        raise NotImplementedError

    @abstractmethod
    def process_demos(self, demo):
        # Must be implemented to handle data augmentation or formatting for demonstrations
        raise NotImplementedError

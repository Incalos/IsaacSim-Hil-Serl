import os
from abc import abstractmethod
import yaml
from pathlib import Path


class TrainConfig:
    """Training configuration class for robot environment setup.

    Manages loading experiment parameters, file paths, and environment configuration.
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
    
    @abstractmethod
    def get_environment(self, fake_env=False, classifier=False):
        # Must be implemented to return the specific gym environment instance
        raise NotImplementedError
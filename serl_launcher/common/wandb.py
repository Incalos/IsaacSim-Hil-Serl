import datetime
import tempfile
import absl.flags as flags
import ml_collections
import wandb
from copy import copy
from socket import gethostname
from typing import List, Tuple, Optional


def _recursive_flatten_dict(d: dict) -> Tuple[List[str], List]:
    """Recursively flatten nested dictionary into flat key-value lists.

    Args:
        d: Nested dictionary to flatten

    Returns:
        Tuple of (flat_keys, flat_values) where keys use '/' for nested levels
    """
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    """Logger class for Weights & Biases (WandB) experiment tracking.

    Provides standardized setup and logging for WandB runs with flat nested dictionaries.
    """

    @staticmethod
    def get_default_config() -> ml_collections.ConfigDict:
        """Get default configuration for WandBLogger.

        Returns:
            Default config with project name, entity, and experiment identifiers
        """
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        config.exp_descriptor = ""
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(
        self,
        wandb_config: ml_collections.ConfigDict,
        variant: dict,
        wandb_output_dir: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """Initialize WandBLogger and setup WandB run.

        Args:
            wandb_config: Configuration for WandB (from get_default_config)
            variant: Experiment hyperparameters/variant to log
            wandb_output_dir: Directory for WandB output files (default: temp directory)
            debug: Whether to disable online logging (debug mode) (default: False)
        """
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.config.experiment_id = self.experiment_id = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        mode = "disabled" if debug else "online"

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            tags=self.config.tag if hasattr(self.config, "tag") else None,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}

        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()

        wandb.config.update(flag_dict)

    def log(self, data: dict, step: Optional[int] = None) -> None:
        """Log flattened dictionary data to WandB with optional step.

        Args:
            data: Nested or flat dictionary of metrics to log
            step: Training/evaluation step number (default: None)
        """
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb.log(data, step=step)

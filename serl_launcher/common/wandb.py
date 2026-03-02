import datetime
import tempfile
from copy import copy
from socket import gethostname
import absl.flags as flags
import ml_collections
import wandb


def _recursive_flatten_dict(d: dict):
    # Initialize lists to store flattened keys and values
    keys, values = [], []

    # Iterate through each key-value pair in the dictionary
    for key, value in d.items():
        # Recursively flatten nested dictionaries
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            # Add non-dict values directly to the lists
            keys.append(key)
            values.append(value)

    return keys, values


class WandBLogger(object):

    @staticmethod
    def get_default_config():
        # Create default configuration for WandB logging
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        config.exp_descriptor = ""
        config.unique_identifier = ""
        config.group = None

        return config

    def __init__(self, wandb_config, variant, wandb_output_dir=None, debug=False):
        # Initialize configuration from input
        self.config = wandb_config

        # Generate unique identifier if not provided
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create unique experiment ID
        self.config.experiment_id = self.experiment_id = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"

        print(self.config)

        # Set default output directory if not provided
        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        # Create a deep copy of the variant dictionary
        self._variant = copy(variant)

        # Add hostname to variant metadata if not present
        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        # Set WandB run mode (disabled for debug, online otherwise)
        mode = "disabled" if debug else "online"

        # Initialize WandB run with configuration
        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            tags=self.config.tag,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        # Handle ABSL flags and update WandB config
        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}

        # Convert ConfigDict values to regular dictionaries
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()

        # Update WandB config with ABSL flags
        wandb.config.update(flag_dict)

    def log(self, data: dict, step: int = None):
        # Flatten nested dictionary data for logging
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}

        # Log flattened data to WandB with optional step number
        wandb.log(data, step=step)

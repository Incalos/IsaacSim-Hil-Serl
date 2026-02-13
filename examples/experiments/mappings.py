# Import specific experiment training configurations with descriptive aliases
from experiments.so101_pick_oranges.config import TrainConfig as SO101PickOrangesTrainConfig

# Registry mapping: Links unique experiment identifiers (strings) to their respective configuration classes.
# This allows the launcher to dynamically instantiate the correct environment/training setup.
CONFIG_MAPPING = {
    "so101_pick_oranges": SO101PickOrangesTrainConfig,
}

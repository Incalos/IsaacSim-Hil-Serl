"""Isaac Lab task package for environments in `robot_infra`."""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

# Auto-import task subpackages to register Gym environments.
import_packages(__name__, _BLACKLIST_PKGS)

# Re-export task modules for convenient imports (e.g. `from robot_infra.isaacsim_venvs import SO101_FOLLOWER_CFG`).
from .so101_pick_oranges import *  # noqa: F401,F403

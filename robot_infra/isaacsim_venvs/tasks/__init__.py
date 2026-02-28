"""Isaac Sim / Isaac Lab task package"""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

# Auto-import task subpackages to register Gym environments.
import_packages(__name__, _BLACKLIST_PKGS)

# Re-export task modules for convenient imports.
from .so101_oranges import *  # noqa: F401,F403

from isaaclab_tasks.utils import import_packages

# Define packages to exclude from auto-import (non-task utility modules)
_BLACKLIST_PKGS = ["utils", ".mdp"]

# Auto-import all task subpackages to register Gym environments with Isaac Lab
import_packages(__name__, _BLACKLIST_PKGS)

from .so101_oranges import *  # noqa: F401,F403

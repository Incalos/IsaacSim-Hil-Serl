"""Isaac Lab task package.

This package auto-imports task subpackages to register Gym environments and
expose their public configuration symbols.
"""

from isaaclab_tasks.utils import import_packages

# Avoid importing internal helper modules that should not register tasks/configs.
_BLACKLIST_PKGS = ["utils", ".mdp"]

# Import all task subpackages in this package for environment registration.
import_packages(__name__, _BLACKLIST_PKGS)

# Re-export task modules for convenient imports.
from .so101_pick_oranges import *

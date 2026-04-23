import isaaclab.sim as sim_utils
from pathlib import Path
from isaaclab.assets import AssetBaseCfg

KITCHEN_WITH_ORANGE_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Scene",
    spawn=sim_utils.UsdFileCfg(usd_path=str(Path(__file__).parent / "kitchen_with_orange" / "scene.usd")),
)
"""SO101 pick-oranges environment assets and Gymnasium registration."""

from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg

SCENES_ROOT = Path(__file__).parent / "assets" / "scenes"
ROBOTS_ROOT = Path(__file__).parent / "assets" / "robots"

KITCHEN_WITH_ORANGE_USD_PATH = str(SCENES_ROOT / "kitchen_with_orange" / "scene.usd")
KITCHEN_WITH_ORANGE_CFG = AssetBaseCfg(spawn=sim_utils.UsdFileCfg(usd_path=KITCHEN_WITH_ORANGE_USD_PATH))

SO101_FOLLOWER_ASSET_PATH = str(ROBOTS_ROOT / "so101_follower.usd")
SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_FOLLOWER_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=400,
            solver_velocity_iteration_count=40,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.2, -0.61, 0.89),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.6,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)

# Register the environment so it can be created via `gym.make("SO101-PickOranges", cfg=...)`.
gym.register(
    id="SO101-PickOranges",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_oranges_env_cfg:PickOrangesEnvCfg"},
)

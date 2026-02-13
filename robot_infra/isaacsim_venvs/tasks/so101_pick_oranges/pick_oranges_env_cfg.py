import torch
from typing import Any
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from scipy.spatial.transform import Rotation as R
from . import KITCHEN_WITH_ORANGE_CFG, KITCHEN_WITH_ORANGE_USD_PATH, SO101_FOLLOWER_CFG, mdp


@configclass
class PickOrangesSceneCfg(InteractiveSceneCfg):
    # Define primary assets: the kitchen environment and the SO101 robot articulation
    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Setup frame transformers to track gripper and jaw positions relative to robot base
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper"),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/jaw", name="jaw", offset=OffsetCfg(pos=(-0.021, -0.070, 0.02))
            ),
        ],
    )
    # Configure the wrist-mounted camera with specific ROS-convention offsets
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.03, 0.06, 0.12),
            rot=(0, -1, 0, 0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=0,
    )
    # Configure a static front-view camera for global scene observation
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=0,
    )
    # Configure a static side-view camera for global scene observation
    side: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/side_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.7, -0.2, 0.2),
            rot=(-0.4055797876726388, 0.5792279653395691, 0.5792279653395693, -0.4055797876726387),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=0,
    )
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # State-based observations including joint dynamics and end-effector pose
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        # Visual observations from the two configured cameras
        wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )
        front = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}
        )
        side = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("side"), "data_type": "rgb", "normalize": False}
        )
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        # Boolean logic terms to track picking and placing status for each orange
        pick_orange001 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange001")})
        put_orange001_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange001"), "plate_cfg": SceneEntityCfg("Plate")},
        )
        pick_orange002 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange002")})
        put_orange002_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange002"), "plate_cfg": SceneEntityCfg("Plate")},
        )
        pick_orange003 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange003")})
        put_orange003_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange003"), "plate_cfg": SceneEntityCfg("Plate")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class EventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    # Sparse reward triggered when the overall task criteria are met
    reward: RewTerm = RewTerm(
        func=mdp.task_done,
        weight=1.0,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )


@configclass
class PickOrangesEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickOrangesSceneCfg = PickOrangesSceneCfg(env_spacing=8.0)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    recorders: RecordTerm = RecordTerm()
    task_description: str = "Pick three oranges and put them into the plate, then reset the arm to rest state."
    dynamic_reset_gripper_effort_limit: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # Configure simulation stepping and viewer positioning
        self.decimation = 1
        self.episode_length_s = 25.0
        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (2.0, -0.5, 1.0)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True
        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        # Extract specific object meshes from the USD file into sub-assets
        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH, self, specific_name_list=["Orange001", "Orange002", "Orange003", "Plate"]
        )
        # Apply domain randomization to object initial positions
        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform("Orange001", pose_range={"x": (0.0, 0.03), "y": (-0.03, 0), "z": (0.0, 0.0)}),
                randomize_object_uniform("Orange002", pose_range={"x": (-1, -1), "y": (-1, -1), "z": (0.0, 0.0)}),
                randomize_object_uniform("Orange003", pose_range={"x": (-1, -1), "y": (-1, -1), "z": (0.0, 0.0)}),
                randomize_object_uniform("Plate", pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}),
            ],
        )

    def use_teleop_device(self, teleop_device: str) -> None:
        # Reconfigure the action manager and physics properties for manual control
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device in ["keyboard", "gamepad"]:
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device: str) -> torch.Tensor:
        # Bridge function to convert raw device input into environment-ready tensors
        return preprocess_device_action(action, teleop_device)

"""Environment configuration for the SO101 pick-oranges Isaac Lab task."""

from __future__ import annotations

from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
from leisaac.utils.domain_randomization import domain_randomization, randomize_camera_uniform, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from . import KITCHEN_WITH_ORANGE_CFG, KITCHEN_WITH_ORANGE_USD_PATH, SO101_FOLLOWER_CFG, mdp


@configclass
class PickOrangesSceneCfg(InteractiveSceneCfg):
    """Scene configuration (assets, sensors, and lights)."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper")],
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
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

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ObservationsCfg:
    """Observation terms grouped by consumer (e.g., policy)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by the policy."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        actions = ObsTerm(func=mdp.last_action)

        wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )
        front = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False},
        )

        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )

        joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the environment.

    Terms are intentionally left unspecified so downstream scripts can enable/override them.
    """

    pass


@configclass
class EventsCfg:
    """Event terms (typically triggered at reset)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ActionsCfg:
    """Action terms for the environment.

    Action configuration is injected dynamically, e.g., when a teleop device is enabled.
    """

    pass


@configclass
class RewardsCfg:
    """Reward terms for the environment."""

    reward: RewTerm = RewTerm(
        func=mdp.task_done,
        weight=1.0,
        params={
            "oranges_cfg": [
                SceneEntityCfg("Orange001"),
                SceneEntityCfg("Orange002"),
                SceneEntityCfg("Orange003"),
            ],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )


@configclass
class PickOrangesEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level configuration for the pick-oranges task."""

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

        self.decimation = 1
        self.episode_length_s = 25.0

        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (2.0, -0.5, 1.0)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH,
            self,
            specific_name_list=["Orange001", "Orange002", "Orange003", "Plate"],
        )

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "Orange001",
                    pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
                ),
                randomize_object_uniform(
                    "Orange002",
                    pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
                ),
                randomize_object_uniform(
                    "Orange003",
                    pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
                ),
                randomize_object_uniform(
                    "Plate",
                    pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.025, 0.025),
                        "y": (-0.025, 0.025),
                        "z": (-0.025, 0.025),
                        "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )

    def use_teleop_device(self, teleop_device: str) -> None:
        """Switch the action pipeline to a teleoperation device."""
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)

        if teleop_device in ["keyboard", "gamepad"]:
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device: str) -> torch.Tensor:
        """Convert device-specific actions into the tensor format expected by the environment."""
        return preprocess_device_action(action, teleop_device)

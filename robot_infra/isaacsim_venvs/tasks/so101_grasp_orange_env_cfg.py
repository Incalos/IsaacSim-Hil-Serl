import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from .scenes import KITCHEN_WITH_ORANGE_CFG
from .robots.so101 import SO101_CFG


@configclass
class GraspOrangeSceneCfg(InteractiveSceneCfg):
    """Configuration for the SO101-Grasp-Orange task scene.

    Includes kitchen/orange scene asset, SO101 robot, multi-view cameras (wrist/front/side) and lighting.

    Attributes:
        scene: Kitchen with orange scene asset configuration
        robot: SO101 robot articulation configuration
        wrist: Wrist-mounted camera configuration
        front: Front camera configuration on robot base
        side: Side camera configuration on robot base
        light: Dome light configuration for scene illumination
    """

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_camera",
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

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.3, 0.03, 0.65),
            rot=(0, 0.7071, 0.7071, 0),
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

    side: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/side_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.3, 0.6, 0.15),
            rot=(0, 0, 0.8192, -0.5736),
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
    """Configuration for environment observations in SO101-Grasp-Orange task.

    Placeholder for observation specs (state/rgb/proprioception).
    """

    pass


@configclass
class TerminationsCfg:
    """Configuration for environment termination conditions in SO101-Grasp-Orange task.

    Placeholder for termination rules (success/failure/time-out).
    """

    pass


@configclass
class ActionsCfg:
    """Configuration for environment action space in SO101-Grasp-Orange task.

    Placeholder for action specs (joint control/gripper commands).
    """

    pass


@configclass
class RewardsCfg:
    """Configuration for environment reward functions in SO101-Grasp-Orange task.

    Placeholder for reward shaping (distance-to-orange/action-cost).
    """

    pass


@configclass
class EventsCfg:
    """Configuration for environment event handlers in SO101-Grasp-Orange task.

    Placeholder for reset/randomization events.
    """

    pass


@configclass
class GraspOrangeEnvCfg(ManagerBasedRLEnvCfg):
    """Main configuration for SO101-Grasp-Orange RL environment.

    Integrates scene/action/reward/observation configs and sets simulation parameters.

    Attributes:
        scene: Scene configuration for SO101-Grasp-Orange task
        actions: Action space configuration
        rewards: Reward function configuration
        events: Event handler configuration
        observations: Observation specification configuration
        terminations: Termination condition configuration
    """

    scene: GraspOrangeSceneCfg = GraspOrangeSceneCfg(env_spacing=8.0)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post-initialization for environment configuration.

        Configures viewer settings, simulation timestep, PhysX parameters,
        extracts USD sub-assets and applies domain randomization.
        """
        super().__post_init__()

        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (2.0, -0.5, 1.0)

        self.sim.dt = 1.0 / 240.0
        self.decimation = 2
        self.episode_length_s = 25.0

        self.sim.physx.rest_offset = 0.001
        self.sim.physx.contact_offset = 0.025
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset = 0.001
        self.sim.physx.friction_correlation_distance = 0.0005
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physx.enable_ccd = True

        parse_usd_and_create_subassets(KITCHEN_WITH_ORANGE_CFG.spawn.usd_path, self, specific_name_list=["Orange001"])

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform("Orange001", pose_range={"x": (-0.06, -0.06), "y": (0, 0), "z": (0, 0)})
            ],
        )

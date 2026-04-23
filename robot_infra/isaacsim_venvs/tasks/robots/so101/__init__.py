import isaaclab.sim as sim_utils
from pathlib import Path
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(Path(__file__).parent / "model" / "so101.urdf"),
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1200.0,
                damping=50.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.5,
            angular_damping=0.5,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.2, -0.61, 0.92),   
        rot=(0.70711, 0.0, 0.0, 0.70711),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "load_bearing_joints": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift", "elbow_flex"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=55.0,
            friction=2.0,
            armature=0.01,
        ),
        "non_load_joints": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "wrist_flex", "wrist_roll"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=800.0,
            damping=40.0,
            friction=2.0,
            armature=0.01,
        ),
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=400.0,
            damping=25.0,
            friction=1.0,
            armature=0.005,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)

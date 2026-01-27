"""
SO101 Follower Robot Virtual Environment with ROS2 Communication.

This module creates a simulation environment for the SO101 follower robotic arm
in Isaac Sim. It provides:
- End-effector inverse kinematics control using cuRobo GPU-accelerated solver
- Joint position control via ROS2 topics
- Camera streaming (wrist and front cameras) through ROS2
- OmniGraph-based communication pipeline for real-time robot control
"""

import os
import sys

# Add project root to Python path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from isaaclab.app import AppLauncher

# Command Line Arguments
parser = argparse.ArgumentParser(description="OmniGraph robot control for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="SO101-PickOranges", help="Name of the task.")
parser.add_argument("--ros2_namespace", type=str, default="so101", help="ROS2 namespace for topics.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Initialize Isaac Sim application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Module Imports (after Isaac Sim initialization)
import omni.graph.core as og
import omni.kit.app
import omni.appwindow
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import usdrt.Sdf
from robot_envs.isaacsim_envs import *  # noqa: F401
import builtins
from isaacsim.core.prims import Articulation
import torch
import carb
import carb.input

# cuRobo imports for GPU-accelerated inverse kinematics
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# Global Variables for OmniGraph Script Node Access
builtins._GLOBAL_IK_SOLVER = None
builtins._GLOBAL_TENSOR_ARGS = None
builtins._GLOBAL_GRAPH_PATH = None
builtins._GLOBAL_JOINT_NAMES = None
builtins._GLOBAL_ROBOT_PRIM = None
builtins._GLOBAL_ROBOT_ASSET = None

# End-Effector IK Script for OmniGraph Script Node
EEF_IK_SCRIPT = """
import numpy as np
import torch
import omni.graph.core as og
import builtins
from curobo.types.math import Pose


def setup(db: og.Database):
    state = db.per_instance_state
    state.ik_solver = getattr(builtins, "_GLOBAL_IK_SOLVER", None)
    state.tensor_args = getattr(builtins, "_GLOBAL_TENSOR_ARGS", None)
    state.graph_path = getattr(builtins, "_GLOBAL_GRAPH_PATH", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)
    if state.joint_names:
        state.num_joints = len(state.joint_names)
    else:
        state.num_joints = 0

    # Preallocate fixed-shape buffers to avoid per-tick allocations.
    # These buffers are kept on the same device/dtype as cuRobo expects.
    if state.tensor_args is not None and hasattr(state.tensor_args, "device"):
        device = state.tensor_args.device
        dtype = state.tensor_args.dtype
        state.pos_buf = torch.zeros((1, 3), device=device, dtype=dtype)
        state.quat_buf = torch.zeros((1, 4), device=device, dtype=dtype)


def compute(db: og.Database):
    state = db.per_instance_state
    # Validate required dependencies are ready.
    if state.ik_solver is None or state.graph_path is None or state.tensor_args is None:
        return
    if not hasattr(state, "pos_buf") or not hasattr(state, "quat_buf"):
        return
    eef_attr = og.Controller.attribute(state.graph_path + "/eef_cmd_sub.outputs:data")
    eef_data = eef_attr.get()
    if eef_data is None or len(eef_data) < 8:
        return
    # Disable autograd for inference-only execution.
    with torch.no_grad():
        # Convert ROS2 array to a temporary tensor (typically CPU),
        # then copy into the preallocated device buffers.
        input_tensor = torch.as_tensor(eef_data, dtype=state.tensor_args.dtype)
        state.pos_buf.copy_(input_tensor[:3].view(1, 3))
        quat_xyzw = input_tensor[3:7]
        state.quat_buf.copy_(quat_xyzw[[3, 0, 1, 2]].view(1, 4))
        goal_pose = Pose(position=state.pos_buf, quaternion=state.quat_buf)
        result = state.ik_solver.solve_single(goal_pose)
        full_commands = np.zeros(state.num_joints, dtype=np.float64)
        if result.success.item():
            # Move the solution from device to host for OmniGraph output.
            solution_np = result.solution.detach().cpu().numpy().flatten()
            full_commands[:5] = solution_np[:5]
            full_commands[5] = eef_data[7]
            db.outputs.joint_names = state.joint_names
            db.outputs.position_command = full_commands
            db.outputs.velocity_command = np.full(state.num_joints, np.nan)
            db.outputs.effort_command = np.full(state.num_joints, np.nan)
        else:
            pass
"""

ROBOT_STATE_SCRIPT = """
import numpy as np
import omni.graph.core as og
import builtins


def setup(db: og.Database):
    # Initialize per-instance state and retrieve global robot prim and joint names
    state = db.per_instance_state
    state.robot = getattr(builtins, "_GLOBAL_ROBOT_PRIM", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)


def compute(db: og.Database):
    state = db.per_instance_state
    # Get measured joint forces and torques (shape: num_joints+1, 6) where first 3 are forces, last 3 are torques
    forces_torques = state.robot.get_measured_joint_forces(joint_names=state.joint_names)
    forces_torques = forces_torques.squeeze(0).detach().cpu().numpy()
    # Get joint velocities for computing end-effector velocity via Jacobian
    joint_velocities = state.robot.get_joint_velocities(joint_names=state.joint_names)
    joint_velocities = joint_velocities.squeeze(0).detach().cpu().numpy()
    # Get measured joint efforts (torques) for computing end-effector wrench
    joint_efforts = state.robot.get_measured_joint_efforts()
    joint_efforts = joint_efforts.squeeze(0).detach().cpu().numpy()
    # Get Jacobian matrices (shape: num_envs, num_links, 6, num_dofs) and extract end-effector Jacobian
    jacobians = state.robot.get_jacobians().detach().cpu().numpy()
    ee_link_idx = jacobians.shape[1] - 1  # End-effector is typically the last link
    jacobian_ee = jacobians[0, ee_link_idx, :, :]  # Extract 6xnum_dofs Jacobian for end-effector
    # Compute end-effector velocity: v_ee = J_ee @ dq (6D Cartesian velocity: [vx, vy, vz, wx, wy, wz])
    ee_vel = jacobian_ee @ joint_velocities
    # Compute end-effector wrench (force and torque) from joint efforts using Jacobian transpose pseudoinverse
    # F_ee = (J_ee^T)^+ @ tau_joint, where F_ee is 6D wrench [fx, fy, fz, tx, ty, tz]
    jacobian_transpose_pinv = np.linalg.pinv(jacobian_ee.T)
    eef_wrench = jacobian_transpose_pinv @ joint_efforts
    # Output joint forces (first 3 components) and torques (last 3 components) as float32 arrays
    db.outputs.measuredJointForces = forces_torques[:, :3].astype(np.float32, copy=False)
    db.outputs.measuredJointTorques = forces_torques[:, 3:].astype(np.float32, copy=False)
    # Output end-effector velocity [vx, vy, vz, wx, wy, wz], wrench [fx, fy, fz, tx, ty, tz] and Jacobian of end-effector as float32 arrays
    db.outputs.measuredEEFVelocities = ee_vel.astype(np.float32, copy=False)
    db.outputs.measuredEEFWrenches = eef_wrench.astype(np.float32, copy=False)
    db.outputs.measuredEEFJacobians = jacobian_ee.astype(np.float32, copy=False)
"""


# OmniGraph Robot Controller Class
class OmniGraphRobotController:
    """
    Controller class for SO101 robot using OmniGraph and ROS2.

    This class sets up the complete OmniGraph pipeline for robot control:
    - ROS2 publishers for joint states and end-effector states
    - ROS2 subscribers for joint commands and end-effector commands
    - Camera publishers for wrist and front cameras
    - cuRobo IK solver for end-effector control
    """

    def __init__(self, env, ros2_namespace):
        """
        Initialize the OmniGraph robot controller.

        Args:
            env: The Isaac Lab environment instance
            ros2_namespace: ROS2 namespace for all topics
        """
        self.env = env
        self.ros2_namespace = ros2_namespace
        self.graph_path = "/RobotControlGraph"

        # Robot and camera USD paths
        self.robot_path = usdrt.Sdf.Path("/World/envs/env_0/Robot")
        self.wrist_camera_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/gripper/wrist_camera")
        self.front_camera_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/base/front_camera")
        self.end_effector_parent_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/base")
        self.end_effector_target_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/gripper")

        # Initialize controller components
        self._enable_extensions()
        self._setup_robot_graph()
        self._init_global_variables()

    def _enable_extensions(self):
        """Enable required Isaac Sim extensions for ROS2 bridge and core nodes."""
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        for ext in ["isaacsim.ros2.bridge", "isaacsim.core.nodes"]:
            if not ext_manager.is_extension_enabled(ext):
                ext_manager.set_extension_enabled_immediate(ext, True)

    def _init_global_variables(self):
        """Initialize cuRobo IK solver and set global variables for script node access."""
        builtins._GLOBAL_GRAPH_PATH = self.graph_path
        builtins._GLOBAL_ROBOT_ASSET = self.env.unwrapped.scene["robot"]
        builtins._GLOBAL_ROBOT_PRIM = Articulation(prim_paths_expr="/World/envs/env_0/Robot")

        # Initialize CUDA tensor configuration
        tensor_args = TensorDeviceType()

        # Configure robot kinematics from USD file
        robot_cfg_dict = {
            "kinematics": {
                "usd_path": SO101_FOLLOWER_ASSET_PATH,
                "usd_robot_root": "/so101_new_calib",
                "base_link": "base",
                "ee_link": "gripper",
                "use_usd_kinematics": True,
            }
        }

        # Load robot configuration and create IK solver
        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_model=None,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        ik_solver = IKSolver(ik_config)
        print("[INFO]: Warming up cuRobo IK Solver (this may take a few seconds)...")
        dummy_pose = Pose(
            position=torch.tensor([[0.3, 0.0, 0.3]], device=tensor_args.device),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=tensor_args.device),
        )
        for _ in range(3):
            _ = ik_solver.solve_single(dummy_pose)
        torch.cuda.synchronize()
        print("[INFO]: Warm-up complete. Ready for ROS2 commands.")

        builtins._GLOBAL_TENSOR_ARGS = tensor_args
        builtins._GLOBAL_IK_SOLVER = ik_solver
        builtins._GLOBAL_JOINT_NAMES = ik_solver.joint_names
        builtins._GLOBAL_JOINT_NAMES.append("gripper")

        # Setup script node connections
        self._setup_eef_ik_script_node()
        self._setup_robot_state_script_node()

    def _setup_eef_ik_script_node(self):
        """Create output attributes and connections for the end-effector IK script node."""
        script_node_path = f"{self.graph_path}/eef_ik_script"
        script_node = og.Controller.node(script_node_path)

        # Create output attributes for joint commands
        og.Controller.create_attribute(
            script_node,
            "joint_names",
            og.Type(og.BaseDataType.TOKEN, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        og.Controller.create_attribute(
            script_node,
            "position_command",
            og.Type(og.BaseDataType.DOUBLE, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        og.Controller.create_attribute(
            script_node,
            "velocity_command",
            og.Type(og.BaseDataType.DOUBLE, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        og.Controller.create_attribute(
            script_node,
            "effort_command",
            og.Type(og.BaseDataType.DOUBLE, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )

        # Connect subscriber to script node to articulation controller pipeline
        connections = [
            (f"{self.graph_path}/eef_cmd_sub.outputs:execOut", f"{self.graph_path}/eef_ik_script.inputs:execIn"),
            (
                f"{self.graph_path}/eef_ik_script.outputs:joint_names",
                f"{self.graph_path}/eef_art_controller.inputs:jointNames",
            ),
            (
                f"{self.graph_path}/eef_ik_script.outputs:position_command",
                f"{self.graph_path}/eef_art_controller.inputs:positionCommand",
            ),
            (
                f"{self.graph_path}/eef_ik_script.outputs:velocity_command",
                f"{self.graph_path}/eef_art_controller.inputs:velocityCommand",
            ),
            (
                f"{self.graph_path}/eef_ik_script.outputs:effort_command",
                f"{self.graph_path}/eef_art_controller.inputs:effortCommand",
            ),
            (f"{self.graph_path}/eef_ik_script.outputs:execOut", f"{self.graph_path}/eef_art_controller.inputs:execIn"),
        ]
        for src, dst in connections:
            og.Controller.connect(src, dst)

    def _setup_robot_state_script_node(self):
        """Create output attributes and connections for the joint state script node."""
        script_node_path = f"{self.graph_path}/robot_state_script"
        script_node = og.Controller.node(script_node_path)

        # Create output attributes for joint forces and torques
        og.Controller.create_attribute(
            script_node,
            "measuredJointForces",
            og.Type(og.BaseDataType.FLOAT, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        og.Controller.create_attribute(
            script_node,
            "measuredJointTorques",
            og.Type(og.BaseDataType.FLOAT, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        og.Controller.create_attribute(
            script_node,
            "measuredEEFVelocities",
            og.Type(og.BaseDataType.FLOAT, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        og.Controller.create_attribute(
            script_node,
            "measuredEEFWrenches",
            og.Type(og.BaseDataType.FLOAT, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        og.Controller.create_attribute(
            script_node,
            "measuredEEFJacobians",
            og.Type(og.BaseDataType.FLOAT, array_depth=1),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
        )
        # Connect script node to articulation controller pipeline
        connections = [
            (
                f"{self.graph_path}/robot_state_script.outputs:measuredJointForces",
                f"{self.graph_path}/joint_force_pub.inputs:data",
            ),
            (
                f"{self.graph_path}/robot_state_script.outputs:execOut",
                f"{self.graph_path}/joint_force_pub.inputs:execIn",
            ),
            (
                f"{self.graph_path}/robot_state_script.outputs:measuredJointTorques",
                f"{self.graph_path}/joint_torque_pub.inputs:data",
            ),
            (
                f"{self.graph_path}/robot_state_script.outputs:execOut",
                f"{self.graph_path}/joint_torque_pub.inputs:execIn",
            ),
            (
                f"{self.graph_path}/robot_state_script.outputs:measuredEEFVelocities",
                f"{self.graph_path}/eef_vel_pub.inputs:data",
            ),
            (f"{self.graph_path}/robot_state_script.outputs:execOut", f"{self.graph_path}/eef_vel_pub.inputs:execIn"),
            (
                f"{self.graph_path}/robot_state_script.outputs:measuredEEFWrenches",
                f"{self.graph_path}/eef_wrench_pub.inputs:data",
            ),
            (
                f"{self.graph_path}/robot_state_script.outputs:execOut",
                f"{self.graph_path}/eef_wrench_pub.inputs:execIn",
            ),
            (
                f"{self.graph_path}/robot_state_script.outputs:measuredEEFJacobians",
                f"{self.graph_path}/eef_jacobian_pub.inputs:data",
            ),
            (
                f"{self.graph_path}/robot_state_script.outputs:execOut",
                f"{self.graph_path}/eef_jacobian_pub.inputs:execIn",
            ),
        ]
        for src, dst in connections:
            og.Controller.connect(src, dst)

    def _setup_robot_graph(self):
        """Create OmniGraph with ROS2 publishers, subscribers, and articulation controllers."""
        keys = og.Controller.Keys
        og.Controller.edit(
            {
                "graph_path": self.graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                # Create all OmniGraph nodes
                keys.CREATE_NODES: [
                    # Core timing and context nodes
                    ("on_tick", "omni.graph.action.OnPlaybackTick"),
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("read_simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    # Camera nodes for wrist and front cameras
                    ("wrist_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("wrist_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("front_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("front_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    # Joint control nodes
                    ("joint_cmd_sub", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("joint_art_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                    # End-effector control nodes
                    ("eef_cmd_sub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                    ("eef_ik_script", "omni.graph.scriptnode.ScriptNode"),
                    ("eef_art_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                    # Joint state publisher node
                    ("joint_state_pub", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("joint_force_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("joint_torque_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    # End-effector state publisher node
                    ("eef_pose_pub", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("eef_wrench_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("eef_vel_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("eef_jacobian_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    # Robot state publisher script node
                    ("robot_state_script", "omni.graph.scriptnode.ScriptNode"),
                ],
                # Configure node parameters
                keys.SET_VALUES: [
                    # Robot state script node configuration
                    ("robot_state_script.inputs:script", ROBOT_STATE_SCRIPT),
                    # Wrist camera configuration
                    ("wrist_render_product.inputs:cameraPrim", self.wrist_camera_path),
                    ("wrist_render_product.inputs:height", 480),
                    ("wrist_render_product.inputs:width", 640),
                    ("wrist_render_product.inputs:enabled", True),
                    ("wrist_camera_helper.inputs:topicName", "wrist_camera/rgb"),
                    ("wrist_camera_helper.inputs:nodeNamespace", self.ros2_namespace),
                    ("wrist_camera_helper.inputs:queueSize", 10),
                    ("wrist_camera_helper.inputs:type", "rgb"),
                    # Front camera configuration
                    ("front_render_product.inputs:cameraPrim", self.front_camera_path),
                    ("front_render_product.inputs:height", 480),
                    ("front_render_product.inputs:width", 640),
                    ("front_render_product.inputs:enabled", True),
                    ("front_camera_helper.inputs:topicName", "front_camera/rgb"),
                    ("front_camera_helper.inputs:nodeNamespace", self.ros2_namespace),
                    ("front_camera_helper.inputs:queueSize", 10),
                    ("front_camera_helper.inputs:type", "rgb"),
                    # End-effector state publisher configuration
                    ("eef_pose_pub.inputs:topicName", "eef_poses"),
                    ("eef_pose_pub.inputs:nodeNamespace", self.ros2_namespace),
                    ("eef_pose_pub.inputs:queueSize", 10),
                    ("eef_pose_pub.inputs:parentPrim", self.end_effector_parent_path),
                    ("eef_pose_pub.inputs:targetPrims", [self.end_effector_target_path]),
                    ("eef_wrench_pub.inputs:topicName", "eef_wrenches"),
                    ("eef_wrench_pub.inputs:nodeNamespace", self.ros2_namespace),
                    ("eef_wrench_pub.inputs:queueSize", 10),
                    ("eef_wrench_pub.inputs:messagePackage", "std_msgs"),
                    ("eef_wrench_pub.inputs:messageSubfolder", "msg"),
                    ("eef_wrench_pub.inputs:messageName", "Float32MultiArray"),
                    ("eef_vel_pub.inputs:topicName", "eef_velocities"),
                    ("eef_vel_pub.inputs:nodeNamespace", self.ros2_namespace),
                    ("eef_vel_pub.inputs:queueSize", 10),
                    ("eef_vel_pub.inputs:messagePackage", "std_msgs"),
                    ("eef_vel_pub.inputs:messageSubfolder", "msg"),
                    ("eef_vel_pub.inputs:messageName", "Float32MultiArray"),
                    ("eef_jacobian_pub.inputs:topicName", "eef_jacobians"),
                    ("eef_jacobian_pub.inputs:nodeNamespace", self.ros2_namespace),
                    ("eef_jacobian_pub.inputs:queueSize", 10),
                    ("eef_jacobian_pub.inputs:messagePackage", "std_msgs"),
                    ("eef_jacobian_pub.inputs:messageSubfolder", "msg"),
                    ("eef_jacobian_pub.inputs:messageName", "Float32MultiArray"),
                    # End-effector command subscriber configuration
                    ("eef_cmd_sub.inputs:topicName", "eef_commands"),
                    ("eef_cmd_sub.inputs:nodeNamespace", self.ros2_namespace),
                    ("eef_cmd_sub.inputs:queueSize", 10),
                    ("eef_cmd_sub.inputs:messagePackage", "std_msgs"),
                    ("eef_cmd_sub.inputs:messageSubfolder", "msg"),
                    ("eef_cmd_sub.inputs:messageName", "Float32MultiArray"),
                    ("eef_ik_script.inputs:script", EEF_IK_SCRIPT),
                    ("eef_art_controller.inputs:targetPrim", self.robot_path),
                    # Joint state publisher configuration
                    ("joint_state_pub.inputs:topicName", "joint_states"),
                    ("joint_state_pub.inputs:nodeNamespace", self.ros2_namespace),
                    ("joint_state_pub.inputs:targetPrim", self.robot_path),
                    ("joint_state_pub.inputs:queueSize", 10),
                    ("joint_force_pub.inputs:topicName", "joint_forces"),
                    ("joint_force_pub.inputs:nodeNamespace", self.ros2_namespace),
                    ("joint_force_pub.inputs:queueSize", 10),
                    ("joint_force_pub.inputs:messagePackage", "std_msgs"),
                    ("joint_force_pub.inputs:messageSubfolder", "msg"),
                    ("joint_force_pub.inputs:messageName", "Float32MultiArray"),
                    ("joint_torque_pub.inputs:topicName", "joint_torques"),
                    ("joint_torque_pub.inputs:nodeNamespace", self.ros2_namespace),
                    ("joint_torque_pub.inputs:queueSize", 10),
                    ("joint_torque_pub.inputs:messagePackage", "std_msgs"),
                    ("joint_torque_pub.inputs:messageSubfolder", "msg"),
                    ("joint_torque_pub.inputs:messageName", "Float32MultiArray"),
                    # Joint command subscriber configuration
                    ("joint_cmd_sub.inputs:topicName", "joint_commands"),
                    ("joint_cmd_sub.inputs:nodeNamespace", self.ros2_namespace),
                    ("joint_cmd_sub.inputs:queueSize", 10),
                    ("joint_art_controller.inputs:targetPrim", self.robot_path),
                ],
                # Connect nodes together
                keys.CONNECT: [
                    # Playback tick triggers all subscriber and publisher nodes
                    ("on_tick.outputs:tick", "joint_cmd_sub.inputs:execIn"),
                    ("on_tick.outputs:tick", "joint_state_pub.inputs:execIn"),
                    ("on_tick.outputs:tick", "eef_cmd_sub.inputs:execIn"),
                    ("on_tick.outputs:tick", "eef_pose_pub.inputs:execIn"),
                    ("on_tick.outputs:tick", "wrist_render_product.inputs:execIn"),
                    ("on_tick.outputs:tick", "front_render_product.inputs:execIn"),
                    ("on_tick.outputs:tick", "robot_state_script.inputs:execIn"),
                    # ROS2 context provides communication context to all ROS2 nodes
                    ("ros2_context.outputs:context", "wrist_camera_helper.inputs:context"),
                    ("ros2_context.outputs:context", "front_camera_helper.inputs:context"),
                    ("ros2_context.outputs:context", "eef_cmd_sub.inputs:context"),
                    ("ros2_context.outputs:context", "eef_pose_pub.inputs:context"),
                    ("ros2_context.outputs:context", "eef_wrench_pub.inputs:context"),
                    ("ros2_context.outputs:context", "eef_vel_pub.inputs:context"),
                    ("ros2_context.outputs:context", "eef_jacobian_pub.inputs:context"),
                    ("ros2_context.outputs:context", "joint_cmd_sub.inputs:context"),
                    ("ros2_context.outputs:context", "joint_state_pub.inputs:context"),
                    ("ros2_context.outputs:context", "joint_force_pub.inputs:context"),
                    ("ros2_context.outputs:context", "joint_torque_pub.inputs:context"),
                    # Simulation time to publishers for message timestamps
                    ("read_simulation_time.outputs:simulationTime", "joint_state_pub.inputs:timeStamp"),
                    ("read_simulation_time.outputs:simulationTime", "eef_pose_pub.inputs:timeStamp"),
                    # Camera render product to camera helper connections
                    ("wrist_render_product.outputs:execOut", "wrist_camera_helper.inputs:execIn"),
                    ("wrist_render_product.outputs:renderProductPath", "wrist_camera_helper.inputs:renderProductPath"),
                    ("front_render_product.outputs:renderProductPath", "front_camera_helper.inputs:renderProductPath"),
                    ("front_render_product.outputs:execOut", "front_camera_helper.inputs:execIn"),
                    # Joint command subscriber to articulation controller
                    ("joint_cmd_sub.outputs:jointNames", "joint_art_controller.inputs:jointNames"),
                    ("joint_cmd_sub.outputs:positionCommand", "joint_art_controller.inputs:positionCommand"),
                    ("joint_cmd_sub.outputs:velocityCommand", "joint_art_controller.inputs:velocityCommand"),
                    ("joint_cmd_sub.outputs:effortCommand", "joint_art_controller.inputs:effortCommand"),
                    ("joint_cmd_sub.outputs:execOut", "joint_art_controller.inputs:execIn"),
                ],
            },
        )


# Main Entry Point
def main():
    """Main entry point: creates environment, configures cuRobo IK, and runs simulation loop."""

    # Parse environment configuration from task name
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Initialize OmniGraph controller with ROS2 interface and cuRobo IK
    OmniGraphRobotController(env, ros2_namespace=args_cli.ros2_namespace)

    def on_keyboard_event(event, *args, **kwargs):
        """Keyboard callback: 'R' resets env and zeros all robot joints."""

        # Only handle key press events.
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        # Ignore keys other than 'R'.
        if event.input != carb.input.KeyboardInput.R:
            return True

        env.reset()
        robot = env.unwrapped.scene["robot"]

        zero_pos = torch.zeros_like(robot.data.joint_pos)
        zero_vel = torch.zeros_like(robot.data.joint_vel)

        robot.write_joint_state_to_sim(zero_pos, zero_vel)

        robot.reset()

        simulation_app.update()

        return True

    # Subscribe to keyboard events so that pressing 'R' invokes the reset logic.
    app_window = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # Print startup information
    print("[INFO]: OmniGraph joint control started with ROS2 bridge and cuRobo IK.")
    print(f"[INFO]: ROS2 namespace: {args_cli.ros2_namespace}")
    print(f"[INFO]: Joint commands subscribed from: {args_cli.ros2_namespace}/joint_commands")
    print(f"[INFO]: End-effector commands subscribed from: {args_cli.ros2_namespace}/eef_commands")
    print(f"[INFO]: Wrist camera published to: {args_cli.ros2_namespace}/wrist_camera/rgb")
    print(f"[INFO]: Front camera published to: {args_cli.ros2_namespace}/front_camera/rgb")
    print(f"[INFO]: Joint states published on: {args_cli.ros2_namespace}/joint_states")
    print(f"[INFO]: Joint forces published on: {args_cli.ros2_namespace}/joint_forces")
    print(f"[INFO]: Joint torques published on: {args_cli.ros2_namespace}/joint_torques")
    print(f"[INFO]: End-effector states published to: {args_cli.ros2_namespace}/eef_poses")
    print(f"[INFO]: End-effector wrenches published to: {args_cli.ros2_namespace}/eef_wrenches")
    print(f"[INFO]: End-effector velocities published to: {args_cli.ros2_namespace}/eef_velocities")

    # Run simulation loop
    try:
        while simulation_app.is_running():
            simulation_app.update()
        print("[INFO]: Simulation stopped")
    except KeyboardInterrupt:
        print("[INFO]: Received keyboard interrupt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

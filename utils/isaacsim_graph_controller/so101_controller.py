from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

import builtins
import usdrt.Sdf  # type: ignore[import-not-found]
from isaacsim.core.prims import Articulation  # type: ignore[import-not-found]
import omni.graph.core as og  # type: ignore[import-not-found]
import omni.kit.app  # type: ignore[import-not-found]
import torch
from robot_infra.isaacsim_venvs import SO101_FOLLOWER_ASSET_PATH
from pathlib import Path

# Global variables injected into builtins for script node access.
builtins._GLOBAL_IK_SOLVER = None
builtins._GLOBAL_TENSOR_ARGS = None
builtins._GLOBAL_GRAPH_PATH = None
builtins._GLOBAL_JOINT_NAMES = None
builtins._GLOBAL_ROBOT_PRIM = None
builtins._GLOBAL_ROBOT_ASSET = None


class SO101_OmniGraphController:
    """Controller for SO101 robot using OmniGraph and ROS2."""

    def __init__(self, env, ros2_namespace):
        self.env = env
        self.ros2_namespace = ros2_namespace
        self.graph_path = "/RobotControlGraph"

        # USD prim paths for robot, cameras, and end-effector.
        self.robot_path = usdrt.Sdf.Path("/World/envs/env_0/Robot")
        self.wrist_camera_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/gripper/wrist_camera")
        self.front_camera_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/base/front_camera")
        self.end_effector_parent_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/base")
        self.end_effector_target_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/gripper")

        self._enable_extensions()
        self._setup_robot_graph()
        self._init_global_variables()

    def _enable_extensions(self):
        """Enable required Isaac Sim extensions."""
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        for ext in ["isaacsim.ros2.bridge", "isaacsim.core.nodes"]:
            if not ext_manager.is_extension_enabled(ext):
                ext_manager.set_extension_enabled_immediate(ext, True)

    def _init_global_variables(self):
        """Initialize cuRobo IK solver and expose globals for script nodes."""
        builtins._GLOBAL_GRAPH_PATH = self.graph_path
        builtins._GLOBAL_ROBOT_ASSET = self.env.unwrapped.scene["robot"]
        builtins._GLOBAL_ROBOT_PRIM = Articulation(prim_paths_expr="/World/envs/env_0/Robot")

        tensor_args = TensorDeviceType()

        # Configure robot kinematics from USD.
        robot_cfg_dict = {
            "kinematics": {
                "usd_path": SO101_FOLLOWER_ASSET_PATH,
                "usd_robot_root": "/so101_new_calib",
                "base_link": "base",
                "ee_link": "gripper",
                "use_usd_kinematics": True,
            }
        }

        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_model=None,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        ik_solver = IKSolver(ik_config)

        # Warm up IK solver to initialize CUDA kernels.
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

        self._setup_eef_ik_script_node()
        self._setup_robot_state_script_node()

    def _setup_eef_ik_script_node(self):
        """Create output attributes and connections for end-effector IK script node."""
        script_node_path = f"{self.graph_path}/eef_ik_script"
        script_node = og.Controller.node(script_node_path)

        # Create output attributes for joint commands.
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

        # Connect: subscriber -> script node -> articulation controller.
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
        """Create output attributes and connections for robot state script node."""
        script_node_path = f"{self.graph_path}/robot_state_script"
        script_node = og.Controller.node(script_node_path)

        # Create output attributes for joint forces, torques, and end-effector state.
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

        # Connect script node outputs to ROS2 publishers.
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
                keys.CREATE_NODES: [
                    # Core timing and context.
                    ("on_tick", "omni.graph.action.OnPlaybackTick"),
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("read_simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    # Cameras.
                    ("wrist_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("wrist_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("front_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("front_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    # Joint control.
                    ("joint_cmd_sub", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("joint_art_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                    # End-effector control.
                    ("eef_cmd_sub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                    ("eef_ik_script", "omni.graph.scriptnode.ScriptNode"),
                    ("eef_art_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                    # Joint state publishers.
                    ("joint_state_pub", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("joint_force_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("joint_torque_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    # End-effector state publishers.
                    ("eef_pose_pub", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("eef_wrench_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("eef_vel_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("eef_jacobian_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    # Robot state script node.
                    ("robot_state_script", "omni.graph.scriptnode.ScriptNode"),
                    # Isaac Sim clock publisher.
                    ('isaacsim_clock','isaacsim.ros2.bridge.ROS2PublishClock')
                ],
                keys.SET_VALUES: [
                    # Isaac Sim clock publisher.
                    ("isaacsim_clock.inputs:topicName", "/clock"),
                    # Robot state script node.
                    ("robot_state_script.inputs:usePath", True),
                    (
                        "robot_state_script.inputs:scriptPath",
                        str(Path(__file__).parent.parent / "isaacsim_script_node/get_so101_state_script.py"),
                    ),
                    # Wrist camera.
                    ("wrist_render_product.inputs:cameraPrim", self.wrist_camera_path),
                    ("wrist_render_product.inputs:height", 480),
                    ("wrist_render_product.inputs:width", 640),
                    ("wrist_render_product.inputs:enabled", True),
                    ("wrist_camera_helper.inputs:topicName", "wrist_camera/rgb"),
                    ("wrist_camera_helper.inputs:nodeNamespace", self.ros2_namespace),
                    ("wrist_camera_helper.inputs:queueSize", 10),
                    ("wrist_camera_helper.inputs:type", "rgb"),
                    # Front camera.
                    ("front_render_product.inputs:cameraPrim", self.front_camera_path),
                    ("front_render_product.inputs:height", 480),
                    ("front_render_product.inputs:width", 640),
                    ("front_render_product.inputs:enabled", True),
                    ("front_camera_helper.inputs:topicName", "front_camera/rgb"),
                    ("front_camera_helper.inputs:nodeNamespace", self.ros2_namespace),
                    ("front_camera_helper.inputs:queueSize", 10),
                    ("front_camera_helper.inputs:type", "rgb"),
                    # End-effector state publishers.
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
                    # End-effector command subscriber.
                    ("eef_cmd_sub.inputs:topicName", "eef_commands"),
                    ("eef_cmd_sub.inputs:nodeNamespace", self.ros2_namespace),
                    ("eef_cmd_sub.inputs:queueSize", 10),
                    ("eef_cmd_sub.inputs:messagePackage", "std_msgs"),
                    ("eef_cmd_sub.inputs:messageSubfolder", "msg"),
                    ("eef_cmd_sub.inputs:messageName", "Float32MultiArray"),
                    ("eef_art_controller.inputs:targetPrim", self.robot_path),
                    (
                        "eef_ik_script.inputs:scriptPath",
                        str(Path(__file__).parent.parent / "isaacsim_script_node/get_so101_eef_ik_script.py"),
                    ),
                    ("eef_ik_script.inputs:usePath", True),
                    # Joint state publishers.
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
                    # Joint command subscriber.
                    ("joint_cmd_sub.inputs:topicName", "joint_commands"),
                    ("joint_cmd_sub.inputs:nodeNamespace", self.ros2_namespace),
                    ("joint_cmd_sub.inputs:queueSize", 10),
                    ("joint_art_controller.inputs:targetPrim", self.robot_path),
                ],
                keys.CONNECT: [
                    # Playback tick triggers all nodes.
                    ("on_tick.outputs:tick", "joint_cmd_sub.inputs:execIn"),
                    ("on_tick.outputs:tick", "joint_state_pub.inputs:execIn"),
                    ("on_tick.outputs:tick", "eef_cmd_sub.inputs:execIn"),
                    ("on_tick.outputs:tick", "eef_pose_pub.inputs:execIn"),
                    ("on_tick.outputs:tick", "wrist_render_product.inputs:execIn"),
                    ("on_tick.outputs:tick", "front_render_product.inputs:execIn"),
                    ("on_tick.outputs:tick", "robot_state_script.inputs:execIn"),
                    ("on_tick.outputs:tick", "isaacsim_clock.inputs:execIn"),
                    # ROS2 context to all ROS2 nodes.
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
                    ("ros2_context.outputs:context", "isaacsim_clock.inputs:context"),
                    # Simulation time to publishers.
                    ("read_simulation_time.outputs:simulationTime", "joint_state_pub.inputs:timeStamp"),
                    ("read_simulation_time.outputs:simulationTime", "eef_pose_pub.inputs:timeStamp"),
                    ("read_simulation_time.outputs:simulationTime", "isaacsim_clock.inputs:timeStamp"),
                    # Camera render products to helpers.
                    ("wrist_render_product.outputs:execOut", "wrist_camera_helper.inputs:execIn"),
                    ("wrist_render_product.outputs:renderProductPath", "wrist_camera_helper.inputs:renderProductPath"),
                    ("front_render_product.outputs:renderProductPath", "front_camera_helper.inputs:renderProductPath"),
                    ("front_render_product.outputs:execOut", "front_camera_helper.inputs:execIn"),
                    # Joint command subscriber to controller.
                    ("joint_cmd_sub.outputs:jointNames", "joint_art_controller.inputs:jointNames"),
                    ("joint_cmd_sub.outputs:positionCommand", "joint_art_controller.inputs:positionCommand"),
                    ("joint_cmd_sub.outputs:velocityCommand", "joint_art_controller.inputs:velocityCommand"),
                    ("joint_cmd_sub.outputs:effortCommand", "joint_art_controller.inputs:effortCommand"),
                    ("joint_cmd_sub.outputs:execOut", "joint_art_controller.inputs:execIn"),
                ],
            },
        )

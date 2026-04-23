import builtins
import torch
import omni.graph.core as og
import omni.kit.app
import usdrt.Sdf
from pathlib import Path
from typing import Any, List, Optional
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from isaacsim.core.prims import Articulation

builtins._GLOBAL_TENSOR_ARGS: Optional[TensorDeviceType] = None
builtins._GLOBAL_IK_SOLVER: Optional[IKSolver] = None
builtins._GLOBAL_JOINT_NAMES: Optional[List[str]] = None
builtins._GLOBAL_GRAPH_PATH: Optional[str] = None
builtins._GLOBAL_ROBOT_PRIM: Optional[Articulation] = None
builtins._GLOBAL_UNWRAP_ENV: Optional[Any] = None


class SO101_ROS2_Controller:
    """Controller for SO101 robot using ROS2 bridge in Isaac Sim.

    Args:
        env: Isaac Sim environment instance
        ros2_namespace: ROS2 namespace for robot communication
    """

    def __init__(self, env: Any, ros2_namespace: str) -> None:
        self.env = env
        self.ros2_namespace = ros2_namespace
        self.graph_path = "/RobotControlGraph"

        self.robot_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/root_joint")
        self.wrist_camera_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/gripper_link/wrist_camera")
        self.front_camera_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/base_link/front_camera")
        self.side_camera_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/base_link/side_camera")
        self.end_effector_parent_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/base_link")
        self.end_effector_target_path = usdrt.Sdf.Path("/World/envs/env_0/Robot/gripper_link")

        self._enable_extensions()
        self._setup_robot_graph()
        self._init_global_variables()

    def _enable_extensions(self) -> None:
        """Enable required Isaac Sim extensions for ROS2 bridge and core nodes."""
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        for ext in ["isaacsim.ros2.bridge", "isaacsim.core.nodes"]:
            if not ext_manager.is_extension_enabled(ext):
                ext_manager.set_extension_enabled_immediate(ext, True)

    def _init_global_variables(self) -> None:
        """Initialize global variables for script node access and IK solver setup."""
        builtins._GLOBAL_GRAPH_PATH = self.graph_path
        builtins._GLOBAL_ROBOT_PRIM = Articulation(prim_paths_expr="/World/envs/env_0/Robot")
        builtins._GLOBAL_UNWRAP_ENV = self.env.unwrapped

        tensor_args = TensorDeviceType(device=torch.device("cuda", 0), dtype=torch.float32)
        robot_cfg = RobotConfig.from_basic(
            urdf_path=str(Path(__file__).parent.parent / "model" / "so101.urdf"),
            base_link="base_link",
            ee_link="gripper_link",
            tensor_args=tensor_args,
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            rotation_threshold=0.001,
            position_threshold=0.001,
            num_seeds=32,
            self_collision_check=True,
            self_collision_opt=True,
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

        self._setup_eef_ik_script_node()
        self._setup_robot_state_script_node()

    def _setup_eef_ik_script_node(self) -> None:
        """Setup script node for end-effector IK calculation and joint command output."""
        script_node_path = f"{self.graph_path}/eef_ik_script"
        script_node = og.Controller.node(script_node_path)

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

    def _setup_robot_state_script_node(self) -> None:
        """Setup script node for robot state collection and output to ROS2 publishers."""
        script_node_path = f"{self.graph_path}/robot_state_script"
        script_node = og.Controller.node(script_node_path)

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

    def _setup_robot_graph(self) -> None:
        """Create and configure OmniGraph with ROS2 publishers, subscribers and articulation controllers."""
        keys = og.Controller.Keys
        og.Controller.edit(
            {
                "graph_path": self.graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("on_tick", "omni.graph.action.OnPlaybackTick"),
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("read_simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("wrist_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("wrist_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("front_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("front_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("side_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("side_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("joint_cmd_sub", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("joint_art_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("eef_cmd_sub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                    ("eef_ik_script", "omni.graph.scriptnode.ScriptNode"),
                    ("eef_art_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("joint_state_pub", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("joint_force_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("joint_torque_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("eef_pose_pub", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("eef_wrench_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("eef_vel_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("eef_jacobian_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("robot_state_script", "omni.graph.scriptnode.ScriptNode"),
                    ("isaacsim_clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("isaacsim_reset_sub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                    ("isaacsim_reset_script", "omni.graph.scriptnode.ScriptNode"),
                ],
                keys.SET_VALUES: [
                    ("isaacsim_clock.inputs:topicName", "/clock"),
                    ("robot_state_script.inputs:scriptPath", str(Path(__file__).parent / "get_state_script_node.py")),
                    ("robot_state_script.inputs:usePath", True),
                    ("wrist_render_product.inputs:cameraPrim", self.wrist_camera_path),
                    ("wrist_render_product.inputs:height", 480),
                    ("wrist_render_product.inputs:width", 640),
                    ("wrist_render_product.inputs:enabled", True),
                    ("wrist_camera_helper.inputs:topicName", "wrist_camera/rgb"),
                    ("wrist_camera_helper.inputs:nodeNamespace", self.ros2_namespace),
                    ("wrist_camera_helper.inputs:queueSize", 10),
                    ("wrist_camera_helper.inputs:type", "rgb"),
                    ("front_render_product.inputs:cameraPrim", self.front_camera_path),
                    ("front_render_product.inputs:height", 480),
                    ("front_render_product.inputs:width", 640),
                    ("front_render_product.inputs:enabled", True),
                    ("front_camera_helper.inputs:topicName", "front_camera/rgb"),
                    ("front_camera_helper.inputs:nodeNamespace", self.ros2_namespace),
                    ("front_camera_helper.inputs:queueSize", 10),
                    ("front_camera_helper.inputs:type", "rgb"),
                    ("side_render_product.inputs:cameraPrim", self.side_camera_path),
                    ("side_render_product.inputs:height", 480),
                    ("side_render_product.inputs:width", 640),
                    ("side_render_product.inputs:enabled", True),
                    ("side_camera_helper.inputs:topicName", "side_camera/rgb"),
                    ("side_camera_helper.inputs:nodeNamespace", self.ros2_namespace),
                    ("side_camera_helper.inputs:queueSize", 10),
                    ("side_camera_helper.inputs:type", "rgb"),
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
                    ("eef_cmd_sub.inputs:topicName", "eef_commands"),
                    ("eef_cmd_sub.inputs:nodeNamespace", self.ros2_namespace),
                    ("eef_cmd_sub.inputs:queueSize", 10),
                    ("eef_cmd_sub.inputs:messagePackage", "std_msgs"),
                    ("eef_cmd_sub.inputs:messageSubfolder", "msg"),
                    ("eef_cmd_sub.inputs:messageName", "Float32MultiArray"),
                    ("eef_art_controller.inputs:targetPrim", self.robot_path),
                    ("eef_ik_script.inputs:scriptPath", str(Path(__file__).parent / "get_eef_ik_script_node.py")),
                    ("eef_ik_script.inputs:usePath", True),
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
                    ("joint_cmd_sub.inputs:topicName", "joint_commands"),
                    ("joint_cmd_sub.inputs:nodeNamespace", self.ros2_namespace),
                    ("joint_cmd_sub.inputs:queueSize", 10),
                    ("joint_art_controller.inputs:targetPrim", self.robot_path),
                    ("isaacsim_reset_sub.inputs:topicName", "isaacsim_reset"),
                    ("isaacsim_reset_sub.inputs:nodeNamespace", self.ros2_namespace),
                    ("isaacsim_reset_sub.inputs:queueSize", 10),
                    ("isaacsim_reset_sub.inputs:messagePackage", "std_msgs"),
                    ("isaacsim_reset_sub.inputs:messageSubfolder", "msg"),
                    ("isaacsim_reset_sub.inputs:messageName", "Int8"),
                    (
                        "isaacsim_reset_script.inputs:scriptPath",
                        str(Path(__file__).parent / "reset_isaacsim_script_node.py"),
                    ),
                    ("isaacsim_reset_script.inputs:usePath", True),
                ],
                keys.CONNECT: [
                    ("on_tick.outputs:tick", "joint_cmd_sub.inputs:execIn"),
                    ("on_tick.outputs:tick", "joint_state_pub.inputs:execIn"),
                    ("on_tick.outputs:tick", "eef_cmd_sub.inputs:execIn"),
                    ("on_tick.outputs:tick", "eef_pose_pub.inputs:execIn"),
                    ("on_tick.outputs:tick", "wrist_render_product.inputs:execIn"),
                    ("on_tick.outputs:tick", "front_render_product.inputs:execIn"),
                    ("on_tick.outputs:tick", "side_render_product.inputs:execIn"),
                    ("on_tick.outputs:tick", "robot_state_script.inputs:execIn"),
                    ("on_tick.outputs:tick", "isaacsim_clock.inputs:execIn"),
                    ("on_tick.outputs:tick", "isaacsim_reset_sub.inputs:execIn"),
                    ("on_tick.outputs:tick", "isaacsim_reset_script.inputs:execIn"),
                    ("ros2_context.outputs:context", "wrist_camera_helper.inputs:context"),
                    ("ros2_context.outputs:context", "front_camera_helper.inputs:context"),
                    ("ros2_context.outputs:context", "side_camera_helper.inputs:context"),
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
                    ("ros2_context.outputs:context", "isaacsim_reset_sub.inputs:context"),
                    ("read_simulation_time.outputs:simulationTime", "joint_state_pub.inputs:timeStamp"),
                    ("read_simulation_time.outputs:simulationTime", "eef_pose_pub.inputs:timeStamp"),
                    ("read_simulation_time.outputs:simulationTime", "isaacsim_clock.inputs:timeStamp"),
                    ("wrist_render_product.outputs:execOut", "wrist_camera_helper.inputs:execIn"),
                    ("wrist_render_product.outputs:renderProductPath", "wrist_camera_helper.inputs:renderProductPath"),
                    ("front_render_product.outputs:execOut", "front_camera_helper.inputs:execIn"),
                    ("front_render_product.outputs:renderProductPath", "front_camera_helper.inputs:renderProductPath"),
                    ("side_render_product.outputs:execOut", "side_camera_helper.inputs:execIn"),
                    ("side_render_product.outputs:renderProductPath", "side_camera_helper.inputs:renderProductPath"),
                    ("joint_cmd_sub.outputs:jointNames", "joint_art_controller.inputs:jointNames"),
                    ("joint_cmd_sub.outputs:positionCommand", "joint_art_controller.inputs:positionCommand"),
                    ("joint_cmd_sub.outputs:velocityCommand", "joint_art_controller.inputs:velocityCommand"),
                    ("joint_cmd_sub.outputs:effortCommand", "joint_art_controller.inputs:effortCommand"),
                    ("joint_cmd_sub.outputs:execOut", "joint_art_controller.inputs:execIn"),
                ],
            },
        )

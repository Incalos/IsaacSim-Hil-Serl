"""Flask + ROS2 bridge server for controlling the SO101 robot.

This module provides a small HTTP API (Flask) that publishes commands to ROS2 topics and exposes the latest received
state. It is intended to be used as a lightweight robot-control backend.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pinocchio as pin
import rclpy
import yaml
from absl import app
from cv_bridge import CvBridge
from flask import Flask, Response, abort, jsonify, request
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None  # type: ignore[assignment]


class SO101ROS2Node(Node):
    """ROS2 node that publishes commands and caches the latest received state."""

    def __init__(self):
        super().__init__("SO101ROS2Node")
        self.get_logger().info("SO101ROS2Node initialized")

        # Parameters may already be declared from a ROS2 launch `--params-file`.
        if not self.has_parameter("config_file"):
            self.declare_parameter("config_file", "")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        if not self.has_parameter("namespace"):
            self.declare_parameter("namespace", "so101")

        self.config_file: str = str(self.get_parameter("config_file").value)
        self.use_sim_time: bool = bool(self.get_parameter("use_sim_time").value)
        self.ros2_namespace: str = str(self.get_parameter("namespace").value)

        self.cv_bridge = CvBridge()

        self._load_parameters()
        self.get_logger().info("Parameters loaded")

        # FK (Pinocchio) cache. Prebuilt once then reused for fast workspace checks.
        self._fk_model = None
        self._fk_data = None
        self._fk_ee_frame_id: int | None = None
        self._fk_lock = threading.Lock()
        self._ensure_fk_model()

        # Latest received robot state (updated by ROS2 subscriptions).
        self.joint_positions = np.zeros(self.num_joints, dtype=np.float64)
        self.joint_velocities = np.zeros(self.num_joints, dtype=np.float64)
        self.joint_efforts = np.zeros(self.num_joints, dtype=np.float64)
        self.joint_forces = np.zeros((self.num_joints, 3), dtype=np.float64)
        self.joint_torques = np.zeros((self.num_joints, 3), dtype=np.float64)
        self.eef_poses_quat = np.zeros(7, dtype=np.float64)
        self.eef_poses_euler = np.zeros(6, dtype=np.float64)
        self.eef_forces = np.zeros(3, dtype=np.float64)
        self.eef_torques = np.zeros(3, dtype=np.float64)
        self.eef_velocities = np.zeros(6, dtype=np.float64)
        self.eef_jacobians = np.zeros((self.num_joints, 6), dtype=np.float64)
        self.wrist_camera = None
        self.front_camera = None

        self.joint_cmd_pub = self.create_publisher(JointState, f"/{self.ros2_namespace}/joint_commands", 10)
        self.eef_cmd_pub = self.create_publisher(Float32MultiArray, f"/{self.ros2_namespace}/eef_commands", 10)
        self.joint_state_sub = self.create_subscription(
            JointState, f"/{self.ros2_namespace}/joint_states", self._joint_state_callback, 10
        )
        self.joint_force_sub = self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/joint_forces", self._joint_force_callback, 10
        )
        self.joint_torque_sub = self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/joint_torques", self._joint_torque_callback, 10
        )
        self.eef_pose_sub = self.create_subscription(
            TFMessage, f"/{self.ros2_namespace}/eef_poses", self._eef_pose_callback, 10
        )
        self.eef_wrench_sub = self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/eef_wrenches", self._eef_wrench_callback, 10
        )
        self.eef_vel_sub = self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/eef_velocities", self._eef_vel_callback, 10
        )
        self.eef_jacobian_sub = self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/eef_jacobians", self._eef_jacobian_callback, 10
        )
        self.wrist_camera_sub = self.create_subscription(
            Image, f"/{self.ros2_namespace}/wrist_camera/rgb", self._wrist_camera_callback, 10
        )
        self.front_camera_sub = self.create_subscription(
            Image, f"/{self.ros2_namespace}/front_camera/rgb", self._front_camera_callback, 10
        )

    def _ensure_fk_model(self) -> None:
        if self._fk_model is not None:
            return

        with self._fk_lock:
            if self._fk_model is not None:
                return

            if get_package_share_directory is None:
                raise RuntimeError("ament_index_python is required to locate the installed 'so101_interfaces' package.")
            share_dir = Path(get_package_share_directory("so101_interfaces"))
            urdf_path = share_dir / "urdf" / "so101.urdf"
            self._fk_model = pin.buildModelFromUrdf(str(urdf_path))
            self._fk_data = self._fk_model.createData()

            # Resolve end-effector frame id (prefer the calibrated gripper frame).
            self._fk_ee_frame_id = self._fk_model.getFrameId("so101_new_calib_gripper")
            if self._fk_ee_frame_id >= len(self._fk_model.frames):
                self._fk_ee_frame_id = len(self._fk_model.frames) - 1

            # Precompute mapping from configured joint order -> Pinocchio q indices.
            self._q_indices: list[int] = []
            for name in self.joint_names:
                # Compatibility: YAML may use 'gripper' while URDF uses 'joints_gripper'.
                target_name = "joints_gripper" if name == "gripper" else name
                try:
                    joint_id = self._fk_model.getJointId(target_name)
                    self._q_indices.append(self._fk_model.idx_qs[joint_id])
                except Exception:
                    self.get_logger().warning(f"Joint {target_name} not found in URDF")

    def joints_to_eef_fk(self, joint_positions: Sequence[float]) -> np.ndarray:
        """Compute EEF FK from joint positions.

        Returns a pose as `[x, y, z, qx, qy, qz, qw]` in the base frame.
        """
        model, data = self._fk_model, self._fk_data
        if model is None or data is None or self._fk_ee_frame_id is None:
            self._ensure_fk_model()
            model, data = self._fk_model, self._fk_data
        if model is None or data is None or self._fk_ee_frame_id is None:
            raise RuntimeError("FK model is not initialized.")

        # Build the configuration vector q using precomputed indices for speed.
        q = np.zeros(model.nq, dtype=np.float64)
        for i, val in enumerate(joint_positions):
            if i < len(self._q_indices):
                q[self._q_indices[i]] = val

        # Compute only what we need (FK + frame placements).
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # Extract pose (translation + quaternion).
        oMf = data.oMf[self._fk_ee_frame_id]
        pos = oMf.translation
        quat = Rotation.from_matrix(oMf.rotation).as_quat()  # [qx, qy, qz, qw]
        return np.concatenate([pos, quat])

    def _load_parameters(self):
        """Load configuration from the YAML file specified by `config_file`."""
        with open(self.config_file, "r") as f:
            params = yaml.safe_load(f)
        self.robot_name = params["robot_name"]
        self.flask_url = params["flask_url"]
        self.flask_port = params["flask_port"]
        self.joint_names = params["joint_names"]
        self.num_joints = len(self.joint_names)
        # Target joint configuration used for reset.
        self.reset_joint_positions: list[float] = params.get("reset_joint_positions", [])
        self.min_translation = params["bounding_box"]["min_translation"]
        self.max_translation = params["bounding_box"]["max_translation"]
        self.min_rotation = params["bounding_box"]["min_rotation"]
        self.max_rotation = params["bounding_box"]["max_rotation"]

    def _joint_state_callback(self, msg: JointState) -> None:
        for i, name in enumerate(self.joint_names):
            if name not in msg.name:
                continue
            idx = msg.name.index(name)
            if idx < len(msg.position):
                self.joint_positions[i] = float(msg.position[idx])
            if idx < len(msg.velocity):
                self.joint_velocities[i] = float(msg.velocity[idx])
            if idx < len(msg.effort):
                self.joint_efforts[i] = float(msg.effort[idx])

    def _joint_force_callback(self, msg: Float32MultiArray) -> None:
        self.joint_forces = np.array(msg.data, dtype=np.float64).reshape(self.num_joints, 3)

    def _joint_torque_callback(self, msg: Float32MultiArray) -> None:
        self.joint_torques = np.array(msg.data, dtype=np.float64).reshape(self.num_joints, 3)

    def _eef_pose_callback(self, msg: TFMessage) -> None:
        target_frame = "gripper"
        for transform_stamped in msg.transforms:
            if transform_stamped.child_frame_id == target_frame:
                p = transform_stamped.transform.translation
                q = transform_stamped.transform.rotation
                self.eef_poses_quat = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                self.eef_poses_euler[:3] = np.array([p.x, p.y, p.z])
                self.eef_poses_euler[3:] = Rotation.from_quat(self.eef_poses_quat[3:]).as_euler("zyx", degrees=True)

    def _eef_wrench_callback(self, msg: Float32MultiArray) -> None:
        self.eef_forces = np.array(msg.data[0:3], dtype=np.float64)
        self.eef_torques = np.array(msg.data[3:6], dtype=np.float64)

    def _eef_vel_callback(self, msg: Float32MultiArray) -> None:
        self.eef_velocities = np.array(msg.data, dtype=np.float64)

    def _eef_jacobian_callback(self, msg: Float32MultiArray) -> None:
        self.eef_jacobians = np.array(msg.data, dtype=np.float64).reshape(self.num_joints, 6)

    def _wrist_camera_callback(self, msg: Image) -> None:
        self.wrist_camera = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def _front_camera_callback(self, msg: Image) -> None:
        self.front_camera = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def publish_joint_command(self, positions: Sequence[float]) -> None:
        if len(positions) != self.num_joints:
            self.get_logger().error(f"Expected {self.num_joints} joint positions, got {len(positions)}.")
            return
        # Workspace check using FK from joint angles.
        if not self.check_robot_workspace(positions, "joint"):
            self.get_logger().error(f"Joint pose {list(positions)} is out of workspace (translation bounds).")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(p) for p in positions]
        msg.velocity = []
        msg.effort = []
        self.joint_cmd_pub.publish(msg)

    def publish_eef_command(self, pose: Sequence[float], gripper_state: float) -> None:
        """Publish an EEF command.

        Message format: `[x, y, z, qx, qy, qz, qw, gripper_state]`.
        The input pose can be either:
        - 6D: `[x, y, z, yaw, pitch, roll]` (degrees, zyx), or
        - 7D: `[x, y, z, qx, qy, qz, qw]`.
        """
        pose_list = [float(x) for x in pose]
        if len(pose_list) == 6:
            # Accept Euler angles and convert to quaternion.
            quat = Rotation.from_euler("zyx", pose_list[3:6], degrees=True).as_quat()  # [qx, qy, qz, qw]
            pose_list = pose_list[:3] + quat.tolist()
        elif len(pose_list) != 7:
            self.get_logger().error(f"Expected 6 or 7 pose components, got {len(pose)}.")
            return
        # Workspace check expects [x, y, z, qx, qy, qz, qw] for EEF.
        if not self.check_robot_workspace(pose_list, "eef"):
            self.get_logger().error(f"EEF pose {pose_list} is out of workspace (translation bounds).")
            return
        msg = Float32MultiArray()
        msg.data = pose_list + [float(gripper_state)]
        self.eef_cmd_pub.publish(msg)

    def reset_robot(self) -> None:
        """Reset the robot to the reset joint positions."""
        # Use the joint configuration loaded from the YAML file.
        if hasattr(self, "reset_joint_positions") and self.reset_joint_positions:
            self.publish_joint_command(self.reset_joint_positions)

    def check_robot_workspace(self, pose: Sequence[float], motion_type: str) -> bool:
        """Check if the robot is within the workspace.

        - motion_type == "eef": pose = [x,y,z,qx,qy,qz,qw]
        - motion_type == "joint": pose = joint positions (YAML order), FK will be used to get EEF pose.
        """
        min_t = np.asarray(self.min_translation, dtype=np.float64)
        max_t = np.asarray(self.max_translation, dtype=np.float64)

        if motion_type == "eef":
            eef_pose = np.asarray(pose, dtype=np.float64).flatten()
        elif motion_type == "joint":
            eef_pose = self.joints_to_eef_fk(pose)
        else:
            return False

        # Currently we only enforce translation bounds (axis-aligned bounding box).
        xyz = eef_pose[:3]
        if not (xyz >= min_t).all():
            return False
        if not (xyz <= max_t).all():
            return False
        return True


class SO101Server:
    """High-level wrapper: runs ROS2 spin thread and provides convenience methods."""

    def __init__(self) -> None:
        self.ros2_node: SO101ROS2Node | None = None
        self.ros2_thread: threading.Thread | None = None
        self._init_ros2()

    def _init_ros2(self) -> None:
        """Start ROS2 in a background thread and create the node."""

        def ros2_spin() -> None:
            rclpy.init()
            self.ros2_node = SO101ROS2Node()
            rclpy.spin(self.ros2_node)
            rclpy.shutdown()

        self.ros2_thread = threading.Thread(target=ros2_spin, daemon=True)
        self.ros2_thread.start()

        time.sleep(1.0)
        if self.ros2_node is None:
            raise RuntimeError("Failed to initialize ROS2 node.")

    def _image_to_jpeg_response(self, img: np.ndarray | None, quality: int = 100) -> Response:
        """Encode BGR image as JPEG and return binary response for fast transfer."""
        if img is None:
            abort(503, description="Camera image not available yet")
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return Response(
            buf.tobytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    def move_joints(self, positions: Sequence[float]) -> None:
        """Send a joint-position command."""
        if self.ros2_node is not None:
            self.ros2_node.publish_joint_command(positions)

    def move_eef(self, pose: Sequence[float], gripper_state: float = 0.0) -> None:
        """Send an end-effector pose command with optional gripper state."""
        if self.ros2_node is not None:
            self.ros2_node.publish_eef_command(pose, gripper_state)

    def reset_robot(self) -> None:
        """Reset the robot to the reset joint positions."""
        if self.ros2_node is not None:
            self.ros2_node.reset_robot()

    def get_state(self) -> dict[str, Any] | None:
        """Return the latest cached state or `None` if ROS2 is not ready."""
        if self.ros2_node is None:
            return None

        return {
            "joint_positions": self.ros2_node.joint_positions.tolist(),
            "joint_velocities": self.ros2_node.joint_velocities.tolist(),
            "joint_efforts": self.ros2_node.joint_efforts.tolist(),
            "joint_forces": self.ros2_node.joint_forces.tolist(),
            "joint_torques": self.ros2_node.joint_torques.tolist(),
            "eef_poses_quat": self.ros2_node.eef_poses_quat.tolist(),
            "eef_poses_euler": self.ros2_node.eef_poses_euler.tolist(),
            "eef_forces": self.ros2_node.eef_forces.tolist(),
            "eef_torques": self.ros2_node.eef_torques.tolist(),
            "eef_velocities": self.ros2_node.eef_velocities.tolist(),
            "eef_jacobians": self.ros2_node.eef_jacobians.tolist(),
        }

    def get_wrist_camera(self) -> Response:
        """Return the latest wrist camera JPEG response."""
        if self.ros2_node is None:
            abort(503, description="ROS2 node not ready")
        return self._image_to_jpeg_response(self.ros2_node.wrist_camera)

    def get_front_camera(self) -> Response:
        """Return the latest front camera JPEG response."""
        if self.ros2_node is None:
            abort(503, description="ROS2 node not ready")
        return self._image_to_jpeg_response(self.ros2_node.front_camera)

    def get_joint_positions(self) -> list[float] | None:
        """Return the latest cached joint positions."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_positions.tolist()

    def get_joint_velocities(self) -> list[float] | None:
        """Return the latest cached joint velocities."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_velocities.tolist()

    def get_joint_efforts(self) -> list[float] | None:
        """Return the latest cached joint efforts."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_efforts.tolist()

    def get_joint_forces(self) -> list[list[float]] | None:
        """Return the latest cached joint forces (shape: [num_joints][3])."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_forces.tolist()

    def get_joint_torques(self) -> list[list[float]] | None:
        """Return the latest cached joint torques (shape: [num_joints][3])."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_torques.tolist()

    def get_eef_poses_quat(self) -> list[float] | None:
        """Return the latest cached EEF pose `[x, y, z, qx, qy, qz, qw]`."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_poses_quat.tolist()

    def get_eef_poses_euler(self) -> list[float] | None:
        """Return the latest cached EEF pose `[x, y, z, yaw, pitch, roll]` (degrees, zyx)."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_poses_euler.tolist()

    def get_eef_forces(self) -> list[float] | None:
        """Return the latest cached EEF forces `[fx, fy, fz]`."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_forces.tolist()

    def get_eef_torques(self) -> list[float] | None:
        """Return the latest cached EEF torques `[tx, ty, tz]`."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_torques.tolist()

    def get_eef_velocities(self) -> list[float] | None:
        """Return the latest cached EEF twist `[vx, vy, vz, wx, wy, wz]`."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_velocities.tolist()

    def get_eef_jacobians(self) -> list[list[float]] | None:
        """Return the latest cached EEF Jacobian (shape: [num_joints][6])."""
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_jacobians.tolist()


def main():
    """Start the Flask server and expose the HTTP endpoints."""

    webapp = Flask(__name__)

    robot_server = SO101Server()
    ros2_node = robot_server.ros2_node
    if ros2_node is None:
        raise RuntimeError("ROS2 node is not ready.")

    time.sleep(2.0)

    @webapp.route("/get_joint_positions", methods=["POST"])
    def get_joint_positions():
        return jsonify(robot_server.get_joint_positions())

    @webapp.route("/get_joint_velocities", methods=["POST"])
    def get_joint_velocities():
        return jsonify(robot_server.get_joint_velocities())

    @webapp.route("/get_joint_efforts", methods=["POST"])
    def get_joint_efforts():
        return jsonify(robot_server.get_joint_efforts())

    @webapp.route("/get_joint_forces", methods=["POST"])
    def get_joint_forces():
        return jsonify(robot_server.get_joint_forces())

    @webapp.route("/get_joint_torques", methods=["POST"])
    def get_joint_torques():
        return jsonify(robot_server.get_joint_torques())

    @webapp.route("/get_eef_poses_quat", methods=["POST"])
    def get_eef_poses_quat():
        return jsonify(robot_server.get_eef_poses_quat())

    @webapp.route("/get_eef_poses_euler", methods=["POST"])
    def get_eef_poses_euler():
        return jsonify(robot_server.get_eef_poses_euler())

    @webapp.route("/get_eef_forces", methods=["POST"])
    def get_eef_forces():
        return jsonify(robot_server.get_eef_forces())

    @webapp.route("/get_eef_torques", methods=["POST"])
    def get_eef_torques():
        return jsonify(robot_server.get_eef_torques())

    @webapp.route("/get_eef_velocities", methods=["POST"])
    def get_eef_velocities():
        return jsonify(robot_server.get_eef_velocities())

    @webapp.route("/get_eef_jacobians", methods=["POST"])
    def get_eef_jacobians():
        return jsonify(robot_server.get_eef_jacobians())

    @webapp.route("/get_wrist_camera", methods=["POST"])
    def get_wrist_camera():
        return robot_server.get_wrist_camera()

    @webapp.route("/get_front_camera", methods=["POST"])
    def get_front_camera():
        return robot_server.get_front_camera()

    @webapp.route("/reset_robot", methods=["POST"])
    def reset_robot():
        robot_server.reset_robot()
        return "Robot reset successfully"

    @webapp.route("/move_joints", methods=["POST"])
    def move_joints():
        positions = request.json["joint_pose"]
        robot_server.move_joints(positions)
        return "Joints moved successfully"

    @webapp.route("/move_eef", methods=["POST"])
    def move_eef():
        pose = request.json["eef_pose"]
        gripper_state = request.json["gripper_state"]
        robot_server.move_eef(pose, gripper_state)
        return "EEF moved successfully"

    @webapp.route("/get_state", methods=["POST"])
    def get_state():
        return jsonify(robot_server.get_state())

    webapp.run(host=ros2_node.flask_url, port=ros2_node.flask_port)


if __name__ == "__main__":
    app.run(main)

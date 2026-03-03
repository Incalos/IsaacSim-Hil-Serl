import struct
import threading
import time
from typing import Any, Sequence
from absl import app
import cv2
from cv_bridge import CvBridge
from flask import Flask, Response, abort, jsonify, request
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Int8
from tf2_msgs.msg import TFMessage


class SO101ROS2Node(Node):

    def __init__(self):
        super().__init__("SO101ROS2Node")
        self.get_logger().info("SO101ROS2Node initialized")

        # Declare and get ROS2 parameters
        self.declare_parameter("namespace", "so101")
        self.declare_parameter("flask_url", "127.0.0.1")
        self.declare_parameter("flask_port", 5000)
        self.declare_parameter("joint_names", ["joints"])
        self.declare_parameter("reset_joint_positions", [0.0])
        self.ros2_namespace: str = self.get_parameter("namespace").value
        self.flask_url: str = self.get_parameter("flask_url").value
        self.flask_port: int = self.get_parameter("flask_port").value
        self.joint_names: list[str] = self.get_parameter("joint_names").value
        self.reset_joint_positions: list[float] = self.get_parameter("reset_joint_positions").value
        self.num_joints = len(self.joint_names)

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Initialize robot state buffers
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
        self.side_camera = None

        # Create ROS2 publishers for robot commands
        self.joint_cmd_pub = self.create_publisher(JointState, f"/{self.ros2_namespace}/joint_commands", 10)
        self.eef_cmd_pub = self.create_publisher(Float32MultiArray, f"/{self.ros2_namespace}/eef_commands", 10)
        self.isaacsim_reset_pub = self.create_publisher(Int8, f"/{self.ros2_namespace}/isaacsim_reset", 10)

        # Create ROS2 subscriptions for robot state feedback
        self.joint_state_sub = self.create_subscription(JointState, f"/{self.ros2_namespace}/joint_states", self._joint_state_callback, 10)
        self.joint_force_sub = self.create_subscription(Float32MultiArray, f"/{self.ros2_namespace}/joint_forces", self._joint_force_callback, 10)
        self.joint_torque_sub = self.create_subscription(Float32MultiArray, f"/{self.ros2_namespace}/joint_torques", self._joint_torque_callback, 10)
        self.eef_pose_sub = self.create_subscription(TFMessage, f"/{self.ros2_namespace}/eef_poses", self._eef_pose_callback, 10)
        self.eef_wrench_sub = self.create_subscription(Float32MultiArray, f"/{self.ros2_namespace}/eef_wrenches", self._eef_wrench_callback, 10)
        self.eef_vel_sub = self.create_subscription(Float32MultiArray, f"/{self.ros2_namespace}/eef_velocities", self._eef_vel_callback, 10)
        self.eef_jacobian_sub = self.create_subscription(Float32MultiArray, f"/{self.ros2_namespace}/eef_jacobians", self._eef_jacobian_callback, 10)
        self.wrist_camera_sub = self.create_subscription(Image, f"/{self.ros2_namespace}/wrist_camera/rgb", self._wrist_camera_callback, 10)
        self.front_camera_sub = self.create_subscription(Image, f"/{self.ros2_namespace}/front_camera/rgb", self._front_camera_callback, 10)
        self.side_camera_sub = self.create_subscription(Image, f"/{self.ros2_namespace}/side_camera/rgb", self._side_camera_callback, 10)

    def _joint_state_callback(self, msg: JointState) -> None:
        # Update joint state buffers from incoming JointState message
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
        # Update joint forces buffer (reshape to [num_joints, 3])
        self.joint_forces = np.array(msg.data, dtype=np.float64).reshape(self.num_joints, 3)

    def _joint_torque_callback(self, msg: Float32MultiArray) -> None:
        # Update joint torques buffer (reshape to [num_joints, 3])
        self.joint_torques = np.array(msg.data, dtype=np.float64).reshape(self.num_joints, 3)

    def _eef_pose_callback(self, msg: TFMessage) -> None:
        # Update EEF pose (quaternion + Euler angles) from TF message
        target_frame = "gripper"
        for transform_stamped in msg.transforms:
            if transform_stamped.child_frame_id == target_frame:
                p = transform_stamped.transform.translation
                q = transform_stamped.transform.rotation
                self.eef_poses_quat = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                self.eef_poses_euler[:3] = np.array([p.x, p.y, p.z])
                self.eef_poses_euler[3:] = Rotation.from_quat(self.eef_poses_quat[3:]).as_euler("xyz")

    def _eef_wrench_callback(self, msg: Float32MultiArray) -> None:
        # Update EEF forces and torques buffers
        self.eef_forces = np.array(msg.data[0:3], dtype=np.float64)
        self.eef_torques = np.array(msg.data[3:6], dtype=np.float64)

    def _eef_vel_callback(self, msg: Float32MultiArray) -> None:
        # Update EEF velocities buffer
        self.eef_velocities = np.array(msg.data, dtype=np.float64)

    def _eef_jacobian_callback(self, msg: Float32MultiArray) -> None:
        # Update EEF Jacobian buffer (reshape to [num_joints, 6])
        self.eef_jacobians = np.array(msg.data, dtype=np.float64).reshape(self.num_joints, 6)

    def _wrist_camera_callback(self, msg: Image) -> None:
        # Convert and store wrist camera image (BGR8 format)
        self.wrist_camera = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def _front_camera_callback(self, msg: Image) -> None:
        # Convert and store front camera image (BGR8 format)
        self.front_camera = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def _side_camera_callback(self, msg: Image) -> None:
        # Convert and store side camera image (BGR8 format)
        self.side_camera = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def reset_robot(self) -> None:
        # Reset robot to predefined joint positions and trigger IsaacSim reset
        if hasattr(self, "reset_joint_positions") and self.reset_joint_positions:
            self.publish_joint_command(self.reset_joint_positions)
            msg = Int8()
            msg.data = 1
            self.isaacsim_reset_pub.publish(msg)

    def publish_joint_command(self, positions: Sequence[float]) -> None:
        # Validate and publish joint position command
        if len(positions) != self.num_joints:
            self.get_logger().error(f"Expected {self.num_joints} joint positions, got {len(positions)}.")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(p) for p in positions]
        msg.velocity = []
        msg.effort = []
        self.joint_cmd_pub.publish(msg)

    def publish_eef_command(self, pose: Sequence[float], gripper_state: float) -> None:
        # Process and publish EEF pose command (6D Euler -> 7D quaternion)
        pose_list = [float(x) for x in pose]
        if len(pose_list) == 6:
            quat = Rotation.from_euler("xyz", pose_list[3:6]).as_quat()
            pose_list = pose_list[:3] + quat.tolist()
        elif len(pose_list) != 7:
            self.get_logger().error(f"Expected 6 or 7 pose components, got {len(pose)}.")
            return

        msg = Float32MultiArray()
        msg.data = pose_list + [float(gripper_state)]
        self.eef_cmd_pub.publish(msg)


class SO101Server:

    def __init__(self) -> None:
        self.ros2_node: SO101ROS2Node | None = None
        self.ros2_thread: threading.Thread | None = None
        self._init_ros2()

    def _init_ros2(self) -> None:
        # Initialize ROS2 node in background thread
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

    def move_joints(self, positions: Sequence[float]) -> None:
        # Wrapper for joint command publication
        if self.ros2_node is not None:
            self.ros2_node.publish_joint_command(positions)

    def move_eef(self, pose: Sequence[float], gripper_state: float = 0.0) -> None:
        # Wrapper for EEF command publication
        if self.ros2_node is not None:
            self.ros2_node.publish_eef_command(pose, gripper_state)

    def reset_robot(self) -> None:
        # Wrapper for robot reset functionality
        if self.ros2_node is not None:
            self.ros2_node.reset_robot()

    def get_state(self) -> dict[str, Any] | None:
        # Get consolidated robot state as dictionary
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

    def get_images(self, quality: int | None = None) -> Response:
        # Encode and return all camera images as binary response
        if self.ros2_node is None:
            abort(503, description="ROS2 node not ready")
        names = ["wrist_camera", "front_camera", "side_camera"]
        q = quality if quality is not None else 85
        parts: list[bytes] = []

        for name in names:
            img = getattr(self.ros2_node, name, None)
            if img is None:
                abort(503, description=f"Camera image not available yet: {name}")
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            blob = buf.tobytes()
            parts.append(struct.pack(">I", len(blob)))
            parts.append(blob)

        body = b"".join(parts)
        return Response(
            body,
            mimetype="application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Camera-Names": ",".join(names),
            },
        )

    # Individual state getter methods
    def get_joint_positions(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_positions.tolist()

    def get_joint_velocities(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_velocities.tolist()

    def get_joint_efforts(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_efforts.tolist()

    def get_joint_forces(self) -> list[list[float]] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_forces.tolist()

    def get_joint_torques(self) -> list[list[float]] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.joint_torques.tolist()

    def get_eef_poses_quat(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_poses_quat.tolist()

    def get_eef_poses_euler(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_poses_euler.tolist()

    def get_eef_forces(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_forces.tolist()

    def get_eef_torques(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_torques.tolist()

    def get_eef_velocities(self) -> list[float] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_velocities.tolist()

    def get_eef_jacobians(self) -> list[list[float]] | None:
        if self.ros2_node is None:
            return None
        return self.ros2_node.eef_jacobians.tolist()


def main():
    # Initialize Flask server and define API endpoints
    webapp = Flask(__name__)
    robot_server = SO101Server()
    ros2_node = robot_server.ros2_node

    if ros2_node is None:
        raise RuntimeError("ROS2 node is not ready.")

    # Wait for initial ROS2 data reception
    time.sleep(1.0)

    # Define HTTP endpoints for robot state queries
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

    @webapp.route("/get_images", methods=["POST"])
    def get_images():
        return robot_server.get_images()

    # Define HTTP endpoints for robot control
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

    # Start Flask web server
    webapp.run(host=ros2_node.flask_url, port=ros2_node.flask_port)


if __name__ == "__main__":
    app.run(main)

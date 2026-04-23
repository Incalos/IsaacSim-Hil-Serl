import struct
import threading
import time
import cv2
import numpy as np
import rclpy
import yaml
from absl import app
from cv_bridge import CvBridge
from flask import Flask, Response, abort, jsonify, request
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Int8
from tf2_msgs.msg import TFMessage
from typing import Sequence, Optional, Dict, List, Any
from pathlib import Path


class SO101ROS2Node(Node):
    """ROS2 node for SO101 robot state management and command publishing.

    Manages ROS2 subscriptions for robot state data (joint states, end-effector poses, camera images)
    and publishers for robot commands (joint positions, end-effector poses, reset). Provides thread-safe
    data access for shared robot state and camera image data.
    """

    def __init__(self) -> None:
        super().__init__("SO101ROS2Node")
        self.get_logger().info("SO101ROS2Node initialized!")

        self.declare_parameter("yaml_path", "exp_params.yaml")
        self.yaml_path: str = self.get_parameter("yaml_path").value

        with Path(self.yaml_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.ros2_namespace: str = cfg["namespace"]
        self.flask_url: str = cfg["flask_url"]
        self.flask_port: int = cfg["flask_port"]
        self.joint_names: List[str] = cfg["joint_names"]
        self.reset_joint_positions: List[float] = cfg["reset_joint_positions"]
        self.num_joints = len(self.joint_names)

        self.state_lock = threading.Lock()
        self.image_lock = threading.Lock()

        self.cv_bridge = CvBridge()

        self.joint_positions = [0.0] * self.num_joints
        self.joint_velocities = [0.0] * self.num_joints
        self.joint_efforts = [0.0] * self.num_joints
        self.joint_forces = [[0.0] * 3 for _ in range(self.num_joints)]
        self.joint_torques = [[0.0] * 3 for _ in range(self.num_joints)]
        self.eef_poses_quat = [0.0] * 7
        self.eef_poses_euler = [0.0] * 6
        self.eef_forces = [0.0] * 3
        self.eef_torques = [0.0] * 3
        self.eef_velocities = [0.0] * 6
        self.eef_jacobians = [[0.0] * 6 for _ in range(self.num_joints)]

        self.encoded_images: Dict[str, bytes] = {camera_name: b"" for camera_name in cfg["camera_names"]}
        self.image_quality = 85

        self.joint_cmd_pub = self.create_publisher(JointState, f"/{self.ros2_namespace}/joint_commands", 10)
        self.eef_cmd_pub = self.create_publisher(Float32MultiArray, f"/{self.ros2_namespace}/eef_commands", 10)
        self.isaacsim_reset_pub = self.create_publisher(Int8, f"/{self.ros2_namespace}/isaacsim_reset", 10)

        self.create_subscription(JointState, f"/{self.ros2_namespace}/joint_states", self._joint_state_callback, 10)
        self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/joint_forces", self._joint_force_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/joint_torques", self._joint_torque_callback, 10
        )
        self.create_subscription(TFMessage, f"/{self.ros2_namespace}/eef_poses", self._eef_pose_callback, 10)
        self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/eef_wrenches", self._eef_wrench_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/eef_velocities", self._eef_vel_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, f"/{self.ros2_namespace}/eef_jacobians", self._eef_jacobian_callback, 10
        )

        self.camera_names: List[str] = cfg["camera_names"]
        for camera_name in self.camera_names:
            self.create_subscription(
                Image,
                f"/{self.ros2_namespace}/{camera_name}/rgb",
                lambda m, camera_name=camera_name: self._camera_cb(m, camera_name),
                10,
            )

    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint state ROS2 messages to update joint position/velocity/effort.

        Args:
            msg: JointState message containing joint state data
        """
        with self.state_lock:
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    if idx < len(msg.position):
                        self.joint_positions[i] = float(msg.position[idx])
                    if idx < len(msg.velocity):
                        self.joint_velocities[i] = float(msg.velocity[idx])
                    if idx < len(msg.effort):
                        self.joint_efforts[i] = float(msg.effort[idx])

    def _camera_cb(self, msg: Image, name: str) -> None:
        """Callback for camera image ROS2 messages to convert and encode images to JPEG.

        Args:
            msg: Image message containing raw camera frame
            name: Name of the camera (wrist/front/side) to associate with the image
        """
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            _, buf = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
            blob = buf.tobytes()
            with self.image_lock:
                self.encoded_images[name] = blob
        except Exception as e:
            self.get_logger().error(f"Image compression failed for {name}: {e}")

    def _joint_force_callback(self, msg: Float32MultiArray) -> None:
        """Callback for joint force ROS2 messages to update joint force data.

        Args:
            msg: Float32MultiArray message containing flattened joint force data
        """
        with self.state_lock:
            self.joint_forces = np.array(msg.data).reshape(self.num_joints, 3).tolist()

    def _joint_torque_callback(self, msg: Float32MultiArray) -> None:
        """Callback for joint torque ROS2 messages to update joint torque data.

        Args:
            msg: Float32MultiArray message containing flattened joint torque data
        """
        with self.state_lock:
            self.joint_torques = np.array(msg.data).reshape(self.num_joints, 3).tolist()

    def _eef_pose_callback(self, msg: TFMessage) -> None:
        """Callback for end-effector pose ROS2 messages (TF) to update EEF pose data.

        Converts quaternion pose to Euler angles (xyz) and stores both representations.

        Args:
            msg: TFMessage containing end-effector transform data
        """
        with self.state_lock:
            for ts in msg.transforms:
                if ts.child_frame_id == "gripper_link":
                    p, q = ts.transform.translation, ts.transform.rotation
                    self.eef_poses_quat = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                    euler = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz").tolist()
                    self.eef_poses_euler = [p.x, p.y, p.z] + euler

    def _eef_wrench_callback(self, msg: Float32MultiArray) -> None:
        """Callback for end-effector wrench ROS2 messages to update EEF force/torque data.

        Args:
            msg: Float32MultiArray message containing EEF force (0-2) and torque (3-5) data
        """
        with self.state_lock:
            self.eef_forces = list(msg.data[0:3])
            self.eef_torques = list(msg.data[3:6])

    def _eef_vel_callback(self, msg: Float32MultiArray) -> None:
        """Callback for end-effector velocity ROS2 messages to update EEF velocity data.

        Args:
            msg: Float32MultiArray message containing flattened EEF velocity data
        """
        with self.state_lock:
            self.eef_velocities = list(msg.data)

    def _eef_jacobian_callback(self, msg: Float32MultiArray) -> None:
        """Callback for end-effector jacobian ROS2 messages to update EEF jacobian data.

        Args:
            msg: Float32MultiArray message containing flattened EEF jacobian matrix
        """
        with self.state_lock:
            self.eef_jacobians = np.array(msg.data).reshape(self.num_joints, 6).tolist()

    def reset_robot(self) -> None:
        """Publish joint command to reset robot to predefined joint positions."""
        if hasattr(self, "reset_joint_positions"):
            self.publish_joint_command(self.reset_joint_positions)

    def reset_isaacsim(self) -> None:
        """Publish reset command to Isaac Sim simulation environment."""
        self.isaacsim_reset_pub.publish(Int8(data=1))

    def publish_joint_command(self, positions: Sequence[float]) -> None:
        """Create and publish joint position command to ROS2 topic.

        Args:
            positions: Sequence of target joint positions (one per joint)
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(p) for p in positions]
        self.joint_cmd_pub.publish(msg)

    def publish_eef_command(self, pose: Sequence[float], gripper_state: float = 0.0) -> None:
        """Convert Euler pose to quaternion and publish EEF command to ROS2 topic.

        Args:
            pose: Sequence of EEF pose (xyz + rpy for Euler, xyz + xyzw for quaternion)
            gripper_state: Gripper state value
        """
        pose_list = [float(x) for x in pose]
        if len(pose_list) == 6:
            quat = Rotation.from_euler("xyz", pose_list[3:6]).as_quat()
            pose_list = pose_list[:3] + quat.tolist()
        msg = Float32MultiArray(data=pose_list + [float(gripper_state)])
        self.eef_cmd_pub.publish(msg)


class SO101Server:
    """Flask server wrapper for SO101 ROS2 node to expose robot state/commands via HTTP.

    Manages ROS2 node initialization in a separate thread and provides HTTP endpoints
    for accessing robot state, camera images, and sending control commands.
    """

    def __init__(self) -> None:
        self.ros2_node: Optional[SO101ROS2Node] = None
        self._init_ros2()

    def _init_ros2(self) -> None:
        """Initialize ROS2 node in a daemon thread with MultiThreadedExecutor.

        Waits up to 5 seconds for node initialization to complete.
        """

        def ros2_spin() -> None:
            rclpy.init()
            executor = MultiThreadedExecutor()
            self.ros2_node = SO101ROS2Node()
            executor.add_node(self.ros2_node)
            try:
                executor.spin()
            finally:
                rclpy.shutdown()

        t = threading.Thread(target=ros2_spin, daemon=True)
        t.start()

        timeout = 5.0
        start = time.time()
        while self.ros2_node is None and (time.time() - start) < timeout:
            time.sleep(0.1)

    def get_images(self) -> Response:
        """Return encoded camera images as binary HTTP response.

        Encodes images with length prefix (big-endian 4-byte integer) for each camera.
        Aborts with 503 if node not initialized or images not ready.

        Returns:
            Flask Response with binary image data and camera names header
        """
        if not self.ros2_node:
            abort(503)
        names = self.ros2_node.camera_names
        parts: List[bytes] = []
        with self.ros2_node.image_lock:
            for name in names:
                blob = self.ros2_node.encoded_images[name]
                if not blob:
                    abort(503, description=f"Image {name} not ready")
                parts.append(struct.pack(">I", len(blob)))
                parts.append(blob)
        return Response(b"".join(parts), mimetype="application/octet-stream", headers={"Camera-Names": ",".join(names)})

    def get_state(self) -> Dict[str, Any]:
        """Return complete robot state as dictionary with thread-safe access.

        Returns:
            Dictionary containing all robot state data (joints, EEF, forces, torques, etc.)
        """
        with self.ros2_node.state_lock:
            return {
                "joint_positions": self.ros2_node.joint_positions,
                "joint_velocities": self.ros2_node.joint_velocities,
                "joint_efforts": self.ros2_node.joint_efforts,
                "joint_forces": self.ros2_node.joint_forces,
                "joint_torques": self.ros2_node.joint_torques,
                "eef_poses_quat": self.ros2_node.eef_poses_quat,
                "eef_poses_euler": self.ros2_node.eef_poses_euler,
                "eef_forces": self.ros2_node.eef_forces,
                "eef_torques": self.ros2_node.eef_torques,
                "eef_velocities": self.ros2_node.eef_velocities,
                "eef_jacobians": self.ros2_node.eef_jacobians,
            }

    def get_joint_positions(self) -> List[float]:
        """Get current joint positions with thread-safe access.

        Returns:
            List of current joint position values
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.joint_positions

    def get_joint_velocities(self) -> List[float]:
        """Get current joint velocities with thread-safe access.

        Returns:
            List of current joint velocity values
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.joint_velocities

    def get_joint_efforts(self) -> List[float]:
        """Get current joint efforts with thread-safe access.

        Returns:
            List of current joint effort values
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.joint_efforts

    def get_joint_forces(self) -> List[List[float]]:
        """Get current joint forces with thread-safe access.

        Returns:
            2D list of joint force values (num_joints x 3)
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.joint_forces

    def get_joint_torques(self) -> List[List[float]]:
        """Get current joint torques with thread-safe access.

        Returns:
            2D list of joint torque values (num_joints x 3)
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.joint_torques

    def get_eef_poses_quat(self) -> List[float]:
        """Get end-effector pose (quaternion) with thread-safe access.

        Returns:
            List of EEF pose values (x, y, z, qx, qy, qz, qw)
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.eef_poses_quat

    def get_eef_poses_euler(self) -> List[float]:
        """Get end-effector pose (Euler angles) with thread-safe access.

        Returns:
            List of EEF pose values (x, y, z, roll, pitch, yaw)
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.eef_poses_euler

    def get_eef_forces(self) -> List[float]:
        """Get end-effector forces with thread-safe access.

        Returns:
            List of EEF force values (x, y, z)
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.eef_forces

    def get_eef_torques(self) -> List[float]:
        """Get end-effector torques with thread-safe access.

        Returns:
            List of EEF torque values (x, y, z)
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.eef_torques

    def get_eef_velocities(self) -> List[float]:
        """Get end-effector velocities with thread-safe access.

        Returns:
            List of EEF velocity values
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.eef_velocities

    def get_eef_jacobians(self) -> List[List[float]]:
        """Get end-effector jacobian matrix with thread-safe access.

        Returns:
            2D list of EEF jacobian values (num_joints x 6)
        """
        with self.ros2_node.state_lock:
            return self.ros2_node.eef_jacobians


def main() -> None:
    """Initialize Flask application and register HTTP endpoints for SO101 robot control.

    Starts Flask server with configured host/port from ROS2 node parameters.
    """
    webapp = Flask(__name__)
    robot_server = SO101Server()
    node = robot_server.ros2_node

    @webapp.route("/get_joint_positions", methods=["POST"])
    def get_joint_positions() -> Response:
        """HTTP endpoint to get current joint positions.

        Returns:
            JSON response with joint position values
        """
        return jsonify(robot_server.get_joint_positions())

    @webapp.route("/get_joint_velocities", methods=["POST"])
    def get_joint_velocities() -> Response:
        """HTTP endpoint to get current joint velocities.

        Returns:
            JSON response with joint velocity values
        """
        return jsonify(robot_server.get_joint_velocities())

    @webapp.route("/get_joint_efforts", methods=["POST"])
    def get_joint_efforts() -> Response:
        """HTTP endpoint to get current joint efforts.

        Returns:
            JSON response with joint effort values
        """
        return jsonify(robot_server.get_joint_efforts())

    @webapp.route("/get_joint_forces", methods=["POST"])
    def get_joint_forces() -> Response:
        """HTTP endpoint to get current joint forces.

        Returns:
            JSON response with joint force values
        """
        return jsonify(robot_server.get_joint_forces())

    @webapp.route("/get_joint_torques", methods=["POST"])
    def get_joint_torques() -> Response:
        """HTTP endpoint to get current joint torques.

        Returns:
            JSON response with joint torque values
        """
        return jsonify(robot_server.get_joint_torques())

    @webapp.route("/get_eef_poses_quat", methods=["POST"])
    def get_eef_poses_quat() -> Response:
        """HTTP endpoint to get EEF pose (quaternion).

        Returns:
            JSON response with EEF quaternion pose values
        """
        return jsonify(robot_server.get_eef_poses_quat())

    @webapp.route("/get_eef_poses_euler", methods=["POST"])
    def get_eef_poses_euler() -> Response:
        """HTTP endpoint to get EEF pose (Euler angles).

        Returns:
            JSON response with EEF Euler pose values
        """
        return jsonify(robot_server.get_eef_poses_euler())

    @webapp.route("/get_eef_forces", methods=["POST"])
    def get_eef_forces() -> Response:
        """HTTP endpoint to get EEF forces.

        Returns:
            JSON response with EEF force values
        """
        return jsonify(robot_server.get_eef_forces())

    @webapp.route("/get_eef_torques", methods=["POST"])
    def get_eef_torques() -> Response:
        """HTTP endpoint to get EEF torques.

        Returns:
            JSON response with EEF torque values
        """
        return jsonify(robot_server.get_eef_torques())

    @webapp.route("/get_eef_velocities", methods=["POST"])
    def get_eef_velocities() -> Response:
        """HTTP endpoint to get EEF velocities.

        Returns:
            JSON response with EEF velocity values
        """
        return jsonify(robot_server.get_eef_velocities())

    @webapp.route("/get_eef_jacobians", methods=["POST"])
    def get_eef_jacobians() -> Response:
        """HTTP endpoint to get EEF jacobian matrix.

        Returns:
            JSON response with EEF jacobian values
        """
        return jsonify(robot_server.get_eef_jacobians())

    @webapp.route("/get_images", methods=["POST"])
    def get_images() -> Response:
        """HTTP endpoint to get encoded camera images.

        Returns:
            Binary response with encoded images and camera names header
        """
        return robot_server.get_images()

    @webapp.route("/get_state", methods=["POST"])
    def get_state() -> Response:
        """HTTP endpoint to get complete robot state.

        Returns:
            JSON response with all robot state data
        """
        return jsonify(robot_server.get_state())

    @webapp.route("/reset_robot", methods=["POST"])
    def reset_robot() -> Response:
        """HTTP endpoint to reset robot to predefined joint positions.

        Returns:
            JSON response with success status
        """
        robot_server.ros2_node.reset_robot()
        return jsonify({"status": "success"})

    @webapp.route("/reset_isaacsim", methods=["POST"])
    def reset_isaacsim() -> Response:
        """HTTP endpoint to reset Isaac Sim simulation.

        Returns:
            JSON response with success status
        """
        robot_server.ros2_node.reset_isaacsim()
        return jsonify({"status": "success"})

    @webapp.route("/move_joints", methods=["POST"])
    def move_joints() -> Response:
        """HTTP endpoint to send joint position command.

        Expects JSON body with "joint_pose" key (list of target joint positions).

        Returns:
            JSON response with success status
        """
        robot_server.ros2_node.publish_joint_command(request.json["joint_pose"])
        return jsonify({"status": "success"})

    @webapp.route("/move_eef", methods=["POST"])
    def move_eef() -> Response:
        """HTTP endpoint to send EEF pose command.

        Expects JSON body with "eef_pose" key (list of EEF pose values) and optional "gripper_state".

        Returns:
            JSON response with success status
        """
        robot_server.ros2_node.publish_eef_command(request.json["eef_pose"], request.json["gripper_state"])
        return jsonify({"status": "success"})

    webapp.run(host=node.flask_url, port=node.flask_port, threaded=True, use_reloader=False, debug=False)


if __name__ == "__main__":
    app.run(main)

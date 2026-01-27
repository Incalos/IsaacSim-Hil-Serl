"""
SO101 Robot Control Server with ROS2 Communication.

This file starts a control server for the SO101 robot that communicates
with the robot via ROS2 topics. It provides a Flask API interface similar
to franka_server.py but adapted for SO101 robot with 6 joints + gripper.

The server supports:
- Joint position control (6 joints + gripper, all sent together)
- End-effector position control (with IK)
- State monitoring (joint positions, velocities, end-effector pose, etc.)

Usage:
    python so101_server.py --ros2_namespace=so101 --flask_url=127.0.0.1
"""

from flask import Flask, request, jsonify
import numpy as np
import rclpy
from rclpy.node import Node
import threading
import time
from scipy.spatial.transform import Rotation as R
from absl import app, flags
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import geometry_msgs.msg as geom_msg

FLAGS = flags.FLAGS
flags.DEFINE_string("ros2_namespace", "so101", "ROS2 namespace for topics")
flags.DEFINE_string("flask_url", "127.0.0.1", "URL for the flask server to run on.")
flags.DEFINE_integer("flask_port", 5000, "Port for the flask server to run on.")


class SO101ROS2Node(Node):
    """ROS2 node for SO101 robot communication."""

    def __init__(self, namespace):
        super().__init__("so101_control_api")
        self.namespace = namespace

        # Joint names: 5 arm joints + gripper (total 6)
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.num_joints = len(self.joint_names)

        # Initialize state variables
        self.q = np.zeros(self.num_joints)  # Joint positions (6: 5 arm + wrist_roll + gripper)
        self.dq = np.zeros(self.num_joints)  # Joint velocities
        self.pos = np.zeros(7)  # End-effector pose [x, y, z, qx, qy, qz, qw]
        self.vel = np.zeros(6)  # End-effector velocity [vx, vy, vz, wx, wy, wz]
        self.force = np.zeros(3)  # End-effector force [fx, fy, fz]
        self.torque = np.zeros(3)  # End-effector torque [tx, ty, tz]
        self.jacobian = np.zeros((6, self.num_joints))  # End-effector Jacobian (6x6)

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, f"/{self.namespace}/joint_commands", 10)
        self.eef_cmd_pub = self.create_publisher(Float32MultiArray, f"/{self.namespace}/eef_commands", 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, f"/{self.namespace}/joint_states", self._joint_state_callback, 10
        )

        # Note: End-effector state would come from other topics if available
        # For now, we'll compute it from joint states if needed

        self.get_logger().info(f"SO101 ROS2 node initialized with namespace: {self.namespace}")

    def _joint_state_callback(self, msg):
        """Callback for joint state messages."""
        # Update joint positions and velocities
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                if idx < len(msg.position):
                    self.q[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self.dq[i] = msg.velocity[idx]

        # TODO: Compute end-effector pose from joint positions using forward kinematics
        # For now, we'll need to get this from the robot or compute it
        # This would typically require a robot model/URDF

    def publish_joint_command(self, positions):
        """
        Publish joint position command.

        Args:
            positions: List of 6 joint positions [shoulder_pan, shoulder_lift, elbow_flex,
                      wrist_flex, wrist_roll, gripper]
        """
        if len(positions) != self.num_joints:
            self.get_logger().error(f"Joint command must have {self.num_joints} positions, got {len(positions)}")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(p) for p in positions]
        # For position control, velocity and effort are not used
        msg.velocity = []
        msg.effort = []

        self.joint_cmd_pub.publish(msg)

    def publish_eef_command(self, pose, gripper_state):
        """
        Publish end-effector command.

        Args:
            pose: List of 7 values [x, y, z, qx, qy, qz, qw]
            gripper_state: Gripper position (0-1, where 0 is open, 1 is closed)
        """
        msg = Float32MultiArray()
        # Format: [px, py, pz, qx, qy, qz, qw, gripper_state]
        msg.data = [float(p) for p in pose] + [float(gripper_state)]

        self.eef_cmd_pub.publish(msg)


class SO101Server:
    """Handles SO101 robot control via ROS2."""

    def __init__(self, ros2_namespace):
        self.ros2_namespace = ros2_namespace
        self.ros2_node = None
        self.ros2_thread = None

        # Initialize ROS2 node in a separate thread
        self._init_ros2()

    def _init_ros2(self):
        """Initialize ROS2 node in a separate thread."""

        def ros2_spin():
            rclpy.init()
            self.ros2_node = SO101ROS2Node(self.ros2_namespace)
            rclpy.spin(self.ros2_node)
            rclpy.shutdown()

        self.ros2_thread = threading.Thread(target=ros2_spin, daemon=True)
        self.ros2_thread.start()

        # Wait for node to initialize
        time.sleep(1)

        if self.ros2_node is None:
            raise Exception("Failed to initialize ROS2 node")

    def move_joints(self, positions):
        """
        Move to joint positions.

        Args:
            positions: List of 6 joint positions [shoulder_pan, shoulder_lift, elbow_flex,
                      wrist_flex, wrist_roll, gripper]
        """
        if self.ros2_node is not None:
            self.ros2_node.publish_joint_command(positions)

    def move_eef(self, pose, gripper_state=0.0):
        """
        Move end-effector to pose.

        Args:
            pose: List of 7 values [x, y, z, qx, qy, qz, qw]
            gripper_state: Gripper position (0-1)
        """
        if self.ros2_node is not None:
            self.ros2_node.publish_eef_command(pose, gripper_state)

    def get_state(self):
        """Get current robot state."""
        if self.ros2_node is None:
            return None

        return {
            "q": self.ros2_node.q.tolist(),
            "dq": self.ros2_node.dq.tolist(),
            "pose": self.ros2_node.pos.tolist(),
            "vel": self.ros2_node.vel.tolist(),
            "force": self.ros2_node.force.tolist(),
            "torque": self.ros2_node.torque.tolist(),
            "jacobian": self.ros2_node.jacobian.tolist(),
            "gripper_pos": self.ros2_node.q[-1] if len(self.ros2_node.q) > 0 else 0.0,
        }


###############################################################################


def main(_):
    """Main function to start Flask server."""
    ROS2_NAMESPACE = FLAGS.ros2_namespace
    FLASK_URL = FLAGS.flask_url
    FLASK_PORT = FLAGS.flask_port

    webapp = Flask(__name__)

    # Initialize robot server
    robot_server = SO101Server(ros2_namespace=ROS2_NAMESPACE)

    # Wait a bit for ROS2 node to initialize
    time.sleep(2)

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        state = robot_server.get_state()
        if state:
            return jsonify({"pose": state["pose"]})
        return jsonify({"pose": [0.0] * 7}), 500

    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        state = robot_server.get_state()
        if state and len(state["pose"]) >= 7:
            xyz = state["pose"][:3]
            quat = state["pose"][3:7]
            r = R.from_quat(quat).as_euler("xyz")
            return jsonify({"pose": np.concatenate([xyz, r]).tolist()})
        return jsonify({"pose": [0.0] * 6}), 500

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        state = robot_server.get_state()
        if state:
            return jsonify({"vel": state["vel"]})
        return jsonify({"vel": [0.0] * 6}), 500

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        state = robot_server.get_state()
        if state:
            return jsonify({"force": state["force"]})
        return jsonify({"force": [0.0] * 3}), 500

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        state = robot_server.get_state()
        if state:
            return jsonify({"torque": state["torque"]})
        return jsonify({"torque": [0.0] * 3}), 500

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        state = robot_server.get_state()
        if state:
            return jsonify({"q": state["q"]})
        return jsonify({"q": [0.0] * 6}), 500

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        state = robot_server.get_state()
        if state:
            return jsonify({"dq": state["dq"]})
        return jsonify({"dq": [0.0] * 6}), 500

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        state = robot_server.get_state()
        if state:
            return jsonify({"jacobian": state["jacobian"]})
        return jsonify({"jacobian": [[0.0] * 6] * 6}), 500

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        state = robot_server.get_state()
        if state:
            return jsonify({"gripper": state["gripper_pos"]})
        return jsonify({"gripper": 0.0}), 500

    # Route for Sending joint position command (6 joints together: 5 arm + wrist_roll + gripper)
    @webapp.route("/joints", methods=["POST"])
    def joints():
        """Send joint position command. Expects JSON with 'arr' key containing 6 values.

        Format: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        data = request.json
        if "arr" not in data:
            return "Error: 'arr' key not found in request", 400

        positions = np.array(data["arr"])
        if len(positions) != 6:
            return (
                f"Error: Expected 6 joint positions [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], got {len(positions)}",
                400,
            )

        robot_server.move_joints(positions)
        return "Moved"

    # Route for Sending end-effector pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        """Send end-effector pose command. Expects JSON with 'arr' key containing 7 values [x,y,z,qx,qy,qz,qw]."""
        data = request.json
        if "arr" not in data:
            return "Error: 'arr' key not found in request", 400

        pose = np.array(data["arr"])
        if len(pose) != 7:
            return f"Error: Expected 7 values [x,y,z,qx,qy,qz,qw], got {len(pose)}", 400

        # Extract gripper state if provided, otherwise default to 0
        gripper_state = data.get("gripper", 0.0)

        robot_server.move_eef(pose, gripper_state)
        return "Moved"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        """Open gripper by sending joint command with gripper at 0."""
        state = robot_server.get_state()
        if state:
            current_joints = state["q"].copy()
            current_joints[-1] = 0.0  # Set gripper to 0 (open)
            robot_server.move_joints(current_joints)
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        """Close gripper by sending joint command with gripper at 1."""
        state = robot_server.get_state()
        if state:
            current_joints = state["q"].copy()
            current_joints[-1] = 1.0  # Set gripper to 1 (closed)
            robot_server.move_joints(current_joints)
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        """Move gripper to specific position. Expects JSON with 'gripper_pos' key (0-1)."""
        data = request.json
        if "gripper_pos" not in data:
            return "Error: 'gripper_pos' key not found in request", 400

        gripper_pos = np.clip(float(data["gripper_pos"]), 0.0, 1.0)

        state = robot_server.get_state()
        if state:
            current_joints = state["q"].copy()
            current_joints[-1] = gripper_pos
            robot_server.move_joints(current_joints)

        return "Moved Gripper"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        """Get complete robot state."""
        state = robot_server.get_state()
        if state:
            return jsonify(state)
        return (
            jsonify(
                {
                    "q": [0.0] * 6,
                    "dq": [0.0] * 6,
                    "pose": [0.0] * 7,
                    "vel": [0.0] * 6,
                    "force": [0.0] * 3,
                    "torque": [0.0] * 3,
                    "jacobian": [[0.0] * 6] * 6,
                    "gripper_pos": 0.0,
                }
            ),
            500,
        )

    # Route for Clearing Errors (placeholder, may not be needed for SO101)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        """Clear errors (placeholder for compatibility)."""
        return "Clear"

    print(f"[INFO]: SO101 Robot Server started")
    print(f"[INFO]: ROS2 namespace: {ROS2_NAMESPACE}")
    print(f"[INFO]: Flask server running on {FLASK_URL}:{FLASK_PORT}")
    print(f"[INFO]: Joint commands published to: /{ROS2_NAMESPACE}/joint_commands")
    print(f"[INFO]: End-effector commands published to: /{ROS2_NAMESPACE}/eef_commands")
    print(f"[INFO]: Joint states subscribed from: /{ROS2_NAMESPACE}/joint_states")

    webapp.run(host=FLASK_URL, port=FLASK_PORT)


if __name__ == "__main__":
    app.run(main)

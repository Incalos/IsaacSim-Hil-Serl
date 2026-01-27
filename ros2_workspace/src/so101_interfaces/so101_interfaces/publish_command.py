#!/usr/bin/env python3

"""
ROS2 Joint State Command Publisher for SO101 Robot - Single Home Position

This script publishes a single home position command (all joints at 0)
to control the SO101 robot.

Based on Isaac Sim ROS2 Joint Control tutorial:
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_manipulation.html

Usage:
    python3 publish_command.py

The script will publish one home position command and then exit.

Joint names: ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
"""

import threading
import rclpy
from sensor_msgs.msg import JointState
import time


def main(args=None):
    """Main function for continuous home position publishing."""

    # Initialize ROS2
    rclpy.init(args=args)

    # Create node and publisher
    node = rclpy.create_node('so101_home_publisher')
    pub = node.create_publisher(JointState, '/so101/joint_commands', 10)

    # Joint names for SO101 robot
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    # joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1"]

    # Home position command (all joints at 0)
    home_position = JointState()
    home_position.name = joint_names
    home_position.position = [-0.1, 0.2, 0.4, 0.0, 0.0, 0.0]

    # Spin in a separate thread (following Isaac Sim tutorial pattern)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    # Print startup information
    print("\n=== SO101 Home Position Publisher ===")
    print("Publishing single home position command...")
    print("Joint names:", joint_names)
    print("Home position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
    print("Topic: /so101/joint_commands")
    print("=====================================\n")

    try:
        # Add timestamp to the message
        home_position.header.stamp = node.get_clock().now().to_msg()

        # Publish home position command once
        pub.publish(home_position)

        print("Published Home Position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
        print("Command sent successfully!")

        # Wait a moment to ensure the message is sent
        time.sleep(0.1)

    except Exception as e:
        print(f"Error publishing command: {e}")
    finally:
        rclpy.shutdown()
        thread.join()


if __name__ == '__main__':
    main()
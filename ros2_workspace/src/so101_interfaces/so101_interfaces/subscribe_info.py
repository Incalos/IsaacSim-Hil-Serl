#!/usr/bin/env python3

"""
ROS2 Node to subscribe to joint states and camera RGB images from Isaac Sim.

This node subscribes to:
1. Joint states from /so101/joint_states topic (prints to terminal)
2. Wrist camera RGB image from /so101/wrist_camera/rgb topic
3. Front camera RGB image from /so101/front_camera/rgb topic

All messages use standard ROS2 message formats:
- JointState: sensor_msgs.msg.JointState
- Images: sensor_msgs.msg.Image
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import threading


class InfoSubscriber(Node):
    """ROS2 node that subscribes to joint states and camera images."""

    def __init__(self):
        super().__init__("info_subscriber")

        # ROS2 namespace
        self.namespace = "so101"

        # Create subscribers
        self.joint_state_subscription = self.create_subscription(
            JointState, f"/{self.namespace}/joint_states", self.joint_state_callback, 10
        )

        self.wrist_camera_subscription = self.create_subscription(
            Image, f"/{self.namespace}/wrist_camera/rgb", self.wrist_camera_callback, 10
        )

        self.front_camera_subscription = self.create_subscription(
            Image, f"/{self.namespace}/front_camera/rgb", self.front_camera_callback, 10
        )

        self.get_logger().info("Info subscriber node initialized")
        self.get_logger().info(f"Subscribing to:")
        self.get_logger().info(f"  - Joint states: /{self.namespace}/joint_states")
        self.get_logger().info(f"  - Wrist camera: /{self.namespace}/wrist_camera/rgb")
        self.get_logger().info(f"  - Front camera: /{self.namespace}/front_camera/rgb")

    def joint_state_callback(self, msg):
        """Callback for joint state messages. Prints joint information to terminal."""
        self.get_logger().info("Received joint states:")
        self.get_logger().info(f"  Header: stamp={msg.header.stamp}, frame_id={msg.header.frame_id}")

        if msg.name:
            self.get_logger().info(f"  Joint names: {msg.name}, {type(msg.name[1])}")

        if msg.position:
            positions_str = [f"{pos:.4f}" for pos in msg.position]
            self.get_logger().info(f"  Positions: {positions_str}")

        if msg.velocity:
            velocities_str = [f"{vel:.4f}" for vel in msg.velocity]
            self.get_logger().info(f"  Velocities: {velocities_str}")

        if msg.effort:
            efforts_str = [f"{eff:.4f}" for eff in msg.effort]
            self.get_logger().info(f"  Efforts: {efforts_str}")

        self.get_logger().info("---")

    def wrist_camera_callback(self, msg):
        """Callback for wrist camera RGB images."""
        self.get_logger().info("Received wrist camera image:")
        self.get_logger().info(f"  Header: stamp={msg.header.stamp}, frame_id={msg.header.frame_id}")
        self.get_logger().info(f"  Image dimensions: {msg.width}x{msg.height}")
        self.get_logger().info(f"  Encoding: {msg.encoding}")
        self.get_logger().info(f"  Step: {msg.step}")
        self.get_logger().info(f"  Data size: {len(msg.data)} bytes")
        self.get_logger().info("---")

    def front_camera_callback(self, msg):
        """Callback for front camera RGB images."""
        self.get_logger().info("Received front camera image:")
        self.get_logger().info(f"  Header: stamp={msg.header.stamp}, frame_id={msg.header.frame_id}")
        self.get_logger().info(f"  Image dimensions: {msg.width}x{msg.height}")
        self.get_logger().info(f"  Encoding: {msg.encoding}")
        self.get_logger().info(f"  Step: {msg.step}")
        self.get_logger().info(f"  Data size: {len(msg.data)} bytes")
        self.get_logger().info("---")


def main(args=None):
    """Main function to run the ROS2 node."""
    rclpy.init(args=args)

    # Create the subscriber node
    subscriber_node = InfoSubscriber()

    try:
        # Spin the node to handle callbacks
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        subscriber_node.get_logger().info("Received keyboard interrupt, shutting down...")
    finally:
        # Clean shutdown
        subscriber_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

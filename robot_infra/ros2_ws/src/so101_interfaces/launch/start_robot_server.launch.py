#!/usr/bin/env python3

"""
Launch file for loading parameters from YAML file to ROS2 parameter server.

This launch file starts the parameter_loader node which reads parameters
from a YAML configuration file and uploads them to the ROS2 parameter server.

Usage:
    ros2 launch so101_interfaces load_params.launch.py config_file:=/path/to/config.yaml namespace:=so101
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for parameter loader node."""

    # Default config file path (in package config directory)
    default_config_file = PathJoinSubstitution([FindPackageShare("so101_interfaces"), "config", "so101_params.yaml"])

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config_file,
        description="Path to YAML configuration file containing parameters",
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="so101", description='Namespace prefix for parameters (e.g., "so101")'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Use simulation time if true, otherwise use real time"
    )

    # Get launch configurations
    config_file = LaunchConfiguration("config_file")
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Create SO101 robot server node (Flask + ROS2 bridge)
    robot_server_node = Node(
        package="so101_interfaces",
        executable="robot_server",
        name="so101_robot_server",
        output="screen",
        parameters=[{"config_file": config_file, "namespace": namespace, "use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            config_file_arg,
            namespace_arg,
            use_sim_time_arg,
            robot_server_node,
        ]
    )

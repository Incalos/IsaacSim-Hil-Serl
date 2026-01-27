#!/usr/bin/env python3

"""
Launch file for loading parameters from YAML file to ROS2 parameter server.

This launch file starts the parameter_loader node which reads parameters
from a YAML configuration file and uploads them to the ROS2 parameter server.

Usage:
    ros2 launch so101_interfaces load_params.launch.py config_file:=/path/to/config.yaml namespace:=so101
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for parameter loader node."""
    
    # Default config file path (in package config directory)
    default_config_file = PathJoinSubstitution([
        FindPackageShare('so101_interfaces'),
        'config',
        'so101_params.yaml'
    ])
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to YAML configuration file containing parameters'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='so101',
        description='Namespace prefix for parameters (e.g., "so101")'
    )
    
    # Get launch configurations
    config_file = LaunchConfiguration('config_file')
    namespace = LaunchConfiguration('namespace')
    
    # Create parameter loader node
    parameter_loader_node = Node(
        package='so101_interfaces',
        executable='load_params',
        name='parameter_loader',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'namespace': namespace
        }]
    )
    
    # Log info message
    log_info = LogInfo(
        msg=['Loading parameters from: ', config_file, ' with namespace: ', namespace]
    )
    
    return LaunchDescription([
        config_file_arg,
        namespace_arg,
        log_info,
        parameter_loader_node,
    ])

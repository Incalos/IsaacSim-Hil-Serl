#!/usr/bin/env python3

"""
ROS2 Node to load parameters from YAML file and set them to ROS2 parameter server.

This node reads parameters from a YAML configuration file and uploads them
to the ROS2 parameter server, making them available to other nodes.

Usage:
    ros2 run so101_interfaces load_params --ros-args -p config_file:=/path/to/config.yaml
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import yaml
import os
import sys
from pathlib import Path


class ParameterLoader(Node):
    """ROS2 node that loads parameters from YAML file and sets them to parameter server."""

    def __init__(self):
        super().__init__("parameter_loader")

        # Declare parameter for config file path
        self.declare_parameter("config_file", "")
        self.declare_parameter("namespace", "")
        self.declare_parameter("keep_alive", True)  # Keep node alive so other nodes can query parameters

        # Get config file path from parameter
        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        namespace = self.get_parameter("namespace").get_parameter_value().string_value

        if not config_file:
            self.get_logger().error("No config_file parameter provided!")
            self.get_logger().info("Usage: ros2 run so101_interfaces load_params --ros-args -p config_file:=/path/to/config.yaml")
            raise ValueError("config_file parameter is required")

        # Expand user path and resolve absolute path
        config_file = os.path.expanduser(config_file)
        if not os.path.isabs(config_file):
            # If relative path, try to find it relative to package share directory
            # This handles cases where config_file is passed as a relative path
            try:
                from ament_index_python.packages import get_package_share_directory
                package_share_dir = get_package_share_directory('so101_interfaces')
                potential_path = os.path.join(package_share_dir, config_file)
                if os.path.exists(potential_path):
                    config_file = potential_path
                else:
                    # Try relative to current working directory
                    config_file = os.path.abspath(config_file)
            except ImportError:
                # Fallback to relative path resolution
                config_file = os.path.abspath(config_file)

        if not os.path.exists(config_file):
            self.get_logger().error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.get_logger().info(f"Loading parameters from: {config_file}")

        # Load parameters from YAML file
        try:
            with open(config_file, 'r') as f:
                params_dict = yaml.safe_load(f)
            
            if params_dict is None:
                self.get_logger().warn("YAML file is empty or contains no data")
                return

            # Set parameters to parameter server
            self._set_parameters_recursive(params_dict, namespace)
            
            self.get_logger().info("Successfully loaded all parameters to ROS2 parameter server")

        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing YAML file: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"Error loading parameters: {e}")
            raise

    def _collect_all_parameters(self, params_dict, namespace="", param_list=None):
        """
        Recursively collect all parameter names and values from dictionary.
        
        Args:
            params_dict: Dictionary containing parameters
            namespace: Namespace prefix for parameters (e.g., "so101")
            param_list: List to collect parameters (will be created if None)
        
        Returns:
            List of (name, value) tuples
        """
        if param_list is None:
            param_list = []

        for key, value in params_dict.items():
            param_name = f"{namespace}/{key}" if namespace else key

            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                self._collect_all_parameters(value, param_name, param_list)
            else:
                param_list.append((param_name, value))

        return param_list

    def _set_parameters_recursive(self, params_dict, namespace=""):
        """
        Recursively declare and set parameters from dictionary to ROS2 parameter server.
        
        Args:
            params_dict: Dictionary containing parameters
            namespace: Namespace prefix for parameters (e.g., "so101")
        """
        # First, collect all parameters
        all_params = self._collect_all_parameters(params_dict, namespace)
        
        # Declare all parameters first (using the actual values as defaults)
        # This ensures type matching
        for param_name, value in all_params:
            try:
                # Declare parameter with the actual value as default
                # This ensures the type is correct
                self.declare_parameter(param_name, value)
            except Exception as e:
                self.get_logger().warn(f"Failed to declare parameter {param_name}: {e}")
                # If declaration fails, try to continue anyway
                continue
        
        # Now set all parameter values
        parameters_to_set = []
        for param_name, value in all_params:
            param = self._value_to_parameter(param_name, value)
            if param:
                parameters_to_set.append(param)
                self.get_logger().info(f"Setting parameter: {param_name} = {value} (type: {type(value).__name__})")

        # Set all parameters at once
        if parameters_to_set:
            try:
                result = self.set_parameters(parameters_to_set)
                # Check if any parameters failed to set
                for i, res in enumerate(result):
                    if not res.successful:
                        self.get_logger().error(f"Failed to set parameter {parameters_to_set[i].name}: {res.reason}")
            except Exception as e:
                self.get_logger().error(f"Error setting parameters: {e}")
                raise

    def _value_to_parameter(self, name, value):
        """
        Convert a Python value to ROS2 Parameter object.
        
        Args:
            name: Parameter name
            value: Parameter value (can be bool, int, float, str, list)
        
        Returns:
            Parameter object or None if type is not supported
        """
        if isinstance(value, bool):
            return Parameter(name, Parameter.Type.BOOL, value)
        elif isinstance(value, int):
            return Parameter(name, Parameter.Type.INTEGER, value)
        elif isinstance(value, float):
            return Parameter(name, Parameter.Type.DOUBLE, value)
        elif isinstance(value, str):
            return Parameter(name, Parameter.Type.STRING, value)
        elif isinstance(value, list):
            # Determine list type from first element
            if not value:
                # Empty list defaults to string array
                return Parameter(name, Parameter.Type.STRING_ARRAY, [])
            
            first_elem = value[0]
            if isinstance(first_elem, bool):
                return Parameter(name, Parameter.Type.BOOL_ARRAY, value)
            elif isinstance(first_elem, int):
                return Parameter(name, Parameter.Type.INTEGER_ARRAY, value)
            elif isinstance(first_elem, float):
                return Parameter(name, Parameter.Type.DOUBLE_ARRAY, value)
            elif isinstance(first_elem, str):
                return Parameter(name, Parameter.Type.STRING_ARRAY, value)
            else:
                self.get_logger().warn(f"Unsupported list element type for parameter {name}: {type(first_elem)}")
                return None
        else:
            self.get_logger().warn(f"Unsupported parameter type for {name}: {type(value)}")
            return None


def main(args=None):
    """Main function to run the ROS2 node."""
    rclpy.init(args=args)
    parameter_loader = None

    try:
        # Create the parameter loader node
        parameter_loader = ParameterLoader()
        
        # Check if we should keep the node alive
        keep_alive = parameter_loader.get_parameter("keep_alive").get_parameter_value().bool_value
        
        if keep_alive:
            parameter_loader.get_logger().info("Parameters loaded successfully. Node will keep running to serve parameters.")
            parameter_loader.get_logger().info("Other nodes can query parameters through the parameter service.")
            parameter_loader.get_logger().info("Press Ctrl+C to exit.")
            # Keep node alive so other nodes can query parameters
            rclpy.spin(parameter_loader)
        else:
            # Just set parameters and exit
            rclpy.spin_once(parameter_loader, timeout_sec=0.1)
            parameter_loader.get_logger().info("Parameters loaded successfully. Node will exit.")
        
    except KeyboardInterrupt:
        if parameter_loader:
            parameter_loader.get_logger().info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        # Log error - use node logger if available, otherwise print
        if parameter_loader:
            parameter_loader.get_logger().error(f"Failed to load parameters: {e}")
        else:
            # If node doesn't exist yet, just print the error
            print(f"ERROR: Failed to load parameters: {e}", file=sys.stderr)
    finally:
        # Clean shutdown
        if parameter_loader:
            parameter_loader.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

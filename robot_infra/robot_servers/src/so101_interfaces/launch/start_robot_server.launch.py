from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Resolve the default configuration path dynamically relative to the package installation
    default_config_file = PathJoinSubstitution([FindPackageShare("so101_interfaces"), "config", "so101_params.yaml"])
    # Declare launch arguments to allow CLI overrides for file paths, namespaces, and clock settings
    config_file_arg = DeclareLaunchArgument(
        "config_file", default_value=default_config_file, description="Path to YAML configuration file"
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="so101", description="Namespace prefix for parameters"
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Toggle between simulation and real-time clocks"
    )
    # Reference the declared arguments to be used during node instantiation
    config_file = LaunchConfiguration("config_file")
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    # Define the Robot Server node which bridges Flask requests to ROS2 internal interfaces
    robot_server_node = Node(
        package="so101_interfaces",
        executable="robot_server",
        name="so101_robot_server",
        output="screen",
        parameters=[{"config_file": config_file, "namespace": namespace, "use_sim_time": use_sim_time}],
    )
    # Return the final sequence including argument declarations followed by the node execution
    return LaunchDescription([config_file_arg, namespace_arg, use_sim_time_arg, robot_server_node])

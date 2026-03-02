from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments to allow CLI overrides
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="so101",
        description="Namespace prefix for robot arm parameter configuration and topic isolation",
    )
    flask_url_arg = DeclareLaunchArgument(
        "flask_url",
        default_value="127.0.0.1",
        description="IP address of the Flask backend server for robot arm control communication",
    )
    flask_port_arg = DeclareLaunchArgument(
        "flask_port",
        default_value="5000",
        description="TCP port number of the Flask backend server for robot arm control communication",
    )
    joint_names_arg = DeclareLaunchArgument(
        "joint_names",
        default_value="['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']",
        description="Ordered list of joint names corresponding to the SO101 robot arm kinematic chain",
    )
    reset_joint_positions_arg = DeclareLaunchArgument(
        "reset_joint_positions",
        default_value="[0.0, 0.0, 0.2, 1.0, -1.5708, 1.0]",
        description="Default joint positions (in radians) for SO101 robot arm homing/reset procedure",
    )

    # Define the Robot Server node
    robot_server_node = Node(
        package="so101_interfaces",
        executable="robot_server",
        name="so101_robot_server",
        output="screen",
        parameters=[{
            "allow_undeclared_parameters": True,
            "automatically_declare_parameters_from_overrides": True,
            "use_sim_time": True,
            "namespace": LaunchConfiguration("namespace"),
            "flask_url": LaunchConfiguration("flask_url"),
            "flask_port": LaunchConfiguration("flask_port"),
            "joint_names": LaunchConfiguration("joint_names"),
            "reset_joint_positions": LaunchConfiguration("reset_joint_positions"),
        }],
    )

    # Return the launch description containing arguments and the node
    return LaunchDescription([
        namespace_arg,
        flask_url_arg,
        flask_port_arg,
        joint_names_arg,
        reset_joint_positions_arg,
        robot_server_node,
    ])

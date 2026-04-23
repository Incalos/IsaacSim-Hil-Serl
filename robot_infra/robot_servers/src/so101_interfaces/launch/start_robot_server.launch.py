from pathlib import Path
from typing import List, Any
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def _resolve_params_path(experiment_name: str, params_yaml: str) -> Path:
    """Resolve full path to experiment parameters YAML file.

    Args:
        experiment_name: Name of the experiment folder under examples/experiments/
        params_yaml: Name of the YAML parameter file inside the experiment folder

    Raises:
        ValueError: If experiment_name is empty after stripping whitespace

    Returns:
        Path object pointing to the parameter YAML file
    """
    experiment_name = experiment_name.strip()
    if not experiment_name:
        raise ValueError("Launch arg 'experiment_name' is required (folder under examples/experiments/).")
    yaml_name = params_yaml.strip() or "exp_params.yaml"
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "examples" / "experiments").is_dir():
            return d / "examples" / "experiments" / experiment_name / yaml_name
    raise RuntimeError(f"Could not find repository root (no examples/experiments/) starting from {here}")


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Node]:
    """Set up and configure the robot_server node for launch.

    Resolves parameter file path, validates file existence and content,
    prepares node parameters, and returns the configured Node object.

    Args:
        context: Launch context object for resolving launch configurations
        args: Additional positional arguments (unused)
        kwargs: Additional keyword arguments (unused)

    Raises:
        FileNotFoundError: If the parameter YAML file does not exist
        ValueError: If YAML content is not a dictionary or missing required keys

    Returns:
        List containing the configured robot_server Node object
    """
    params_path = _resolve_params_path(
        LaunchConfiguration("experiment_name").perform(context),
        LaunchConfiguration("params_yaml").perform(context),
    )
    params_path = params_path.resolve()
    if not params_path.is_file():
        raise FileNotFoundError(f"Experiment params YAML not found: {params_path}")
    node_params = {"yaml_path": str(params_path)}
    node_params["allow_undeclared_parameters"] = True
    node_params["automatically_declare_parameters_from_overrides"] = True
    node_params["use_sim_time"] = True
    return [
        Node(
            package="so101_interfaces",
            executable="robot_server",
            name="so101_robot_server",
            output="screen",
            parameters=[node_params],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for robot_server node with configurable arguments.

    Declares launch arguments for experiment name and parameter YAML file,
    and sets up the opaque function to configure the robot_server node.

    Returns:
        Configured LaunchDescription object for robot_server launch
    """
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "experiment_name",
                default_value="",
                description="Experiment folder name under examples/experiments/",
            ),
            DeclareLaunchArgument(
                "params_yaml",
                default_value="exp_params.yaml",
                description="YAML filename inside examples/experiments/<experiment_name>/",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

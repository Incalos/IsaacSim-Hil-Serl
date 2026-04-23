import argparse

parser = argparse.ArgumentParser(description="Start the IsaacSim virtual environment for the SO101 task.")
parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment.")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import yaml
import torch
import gymnasium
import omni.appwindow
from pathlib import Path
from typing import Optional
from isaaclab_tasks.utils import parse_env_cfg
from tasks.robots.so101.ros2_bridge.robot_controller import SO101_ROS2_Controller


def _resolve_params_path(experiment_name: str) -> Path:
    """Resolve full path to experiment parameters YAML file.

    Args:
        experiment_name: Name of the experiment folder under examples/experiments/

    Raises:
        ValueError: If experiment_name is empty after stripping whitespace
        RuntimeError: If repository root with examples/experiments/ is not found

    Returns:
        Path object pointing to the parameter YAML file
    """
    experiment_name = experiment_name.strip()
    if not experiment_name:
        raise ValueError("Launch arg 'experiment_name' is required (folder under examples/experiments/).")
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "examples" / "experiments").is_dir():
            return d / "examples" / "experiments" / experiment_name / "exp_params.yaml"
    raise RuntimeError(f"Could not find repository root (no examples/experiments/) starting from {here}")


def main() -> None:
    """Initialize and run IsaacSim SO101 environment with ROS2 bridge and cuRobo IK integration.

    Handles environment configuration, ROS2 controller initialization, keyboard event registration
    for reset functionality, main simulation loop execution, and resource cleanup.

    Returns:
        None
    """
    params_path = _resolve_params_path(args_cli.experiment_name)
    params_path = params_path.resolve()
    if not params_path.is_file():
        raise FileNotFoundError(f"Experiment params YAML not found: {params_path}")
    with params_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = parse_env_cfg(cfg["task_name"], device="cpu", num_envs=1, use_fabric=False)
    env_cfg.recorders = None
    env_cfg.eval_mode = True
    env: gymnasium.Env = gymnasium.make(cfg["task_name"], cfg=env_cfg)
    env.reset()
    SO101_ROS2_Controller(env, ros2_namespace=cfg["namespace"])

    def on_keyboard_event(event: carb.input.KeyboardEvent, *args, **kwargs) -> bool:
        """Handle keyboard press events to trigger SO101 environment reset.

        Triggers full environment reset with domain randomization when 'R' key is pressed.

        Args:
            event: Keyboard event object with input type and key identifier
            args: Additional positional arguments (unused)
            kwargs: Additional keyword arguments (unused)

        Returns:
            bool: Always returns True to keep event subscription active
        """
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input == carb.input.KeyboardInput.R:
            base_env = env.unwrapped
            all_env_ids: torch.Tensor = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
            base_env.event_manager.apply(
                mode="reset", env_ids=all_env_ids, global_env_step_count=base_env.common_step_counter
            )
        return True

    app_window: Optional[omni.appwindow.IAppWindow] = omni.appwindow.get_default_app_window()
    input_interface: carb.input.IInput = carb.input.acquire_input_interface()
    keyboard: carb.input.IKeyboard = app_window.get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("[INFO]: Launch Isaac Sim for the SO101 specific task.")
    print("[INFO]: Key R = domain randomization.")
    print(f"[INFO]: ROS2 namespace: {cfg['namespace']}")
    print(f"[INFO]: Joint commands subscribed from: {cfg['namespace']}/joint_commands")
    print(f"[INFO]: End-effector commands subscribed from: {cfg['namespace']}/eef_commands")
    print(f"[INFO]: Wrist camera published to: {cfg['namespace']}/wrist_camera/rgb")
    print(f"[INFO]: Front camera published to: {cfg['namespace']}/front_camera/rgb")
    print(f"[INFO]: Side camera published to: {cfg['namespace']}/side_camera/rgb")
    print(f"[INFO]: Joint states published on: {cfg['namespace']}/joint_states")
    print(f"[INFO]: Joint forces published on: {cfg['namespace']}/joint_forces")
    print(f"[INFO]: Joint torques published on: {cfg['namespace']}/joint_torques")
    print(f"[INFO]: End-effector states published to: {cfg['namespace']}/eef_poses")
    print(f"[INFO]: End-effector wrenches published to: {cfg['namespace']}/eef_wrenches")
    print(f"[INFO]: End-effector velocities published to: {cfg['namespace']}/eef_velocities")
    print(f"[INFO]: Environment reset commands published to: {cfg['namespace']}/isaacsim_reset")

    try:
        while simulation_app.is_running():
            simulation_app.update()
        print("[INFO]: Simulation stopped")
    except KeyboardInterrupt:
        print("[INFO]: Received keyboard interrupt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

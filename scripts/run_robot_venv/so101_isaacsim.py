"""SO101 follower robot virtual environment with ROS2 communication."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

parser = argparse.ArgumentParser(description="OmniGraph robot control for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="SO101-PickOranges", help="Name of the task.")
parser.add_argument("--ros2_namespace", type=str, default="so101", help="ROS2 namespace for topics.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.appwindow  # type: ignore[import-not-found]
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg  # type: ignore[import-not-found]
from robot_infra.isaacsim_venvs import *  # noqa: F401
import torch
import carb  # type: ignore[import-not-found]
import carb.input  # type: ignore[import-not-found]
from utils.isaacsim_graph_controller.so101_controller import SO101_OmniGraphController


def main():
    """Create environment, configure cuRobo IK, and run simulation loop."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)

    SO101_OmniGraphController(env, ros2_namespace=args_cli.ros2_namespace)

    def on_keyboard_event(event, *args, **kwargs):
        """Keyboard callback: 'R' resets environment and zeros all robot joints."""
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input != carb.input.KeyboardInput.R:
            return True

        env.reset()
        robot = env.unwrapped.scene["robot"]
        zero_pos = torch.zeros_like(robot.data.joint_pos)
        zero_vel = torch.zeros_like(robot.data.joint_vel)
        robot.write_joint_state_to_sim(zero_pos, zero_vel)
        robot.reset()
        simulation_app.update()
        return True

    # Subscribe to keyboard events for manual reset.
    app_window = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("[INFO]: OmniGraph joint control started with ROS2 bridge and cuRobo IK.")
    print(f"[INFO]: ROS2 namespace: {args_cli.ros2_namespace}")
    print(f"[INFO]: Joint commands subscribed from: {args_cli.ros2_namespace}/joint_commands")
    print(f"[INFO]: End-effector commands subscribed from: {args_cli.ros2_namespace}/eef_commands")
    print(f"[INFO]: Wrist camera published to: {args_cli.ros2_namespace}/wrist_camera/rgb")
    print(f"[INFO]: Front camera published to: {args_cli.ros2_namespace}/front_camera/rgb")
    print(f"[INFO]: Joint states published on: {args_cli.ros2_namespace}/joint_states")
    print(f"[INFO]: Joint forces published on: {args_cli.ros2_namespace}/joint_forces")
    print(f"[INFO]: Joint torques published on: {args_cli.ros2_namespace}/joint_torques")
    print(f"[INFO]: End-effector states published to: {args_cli.ros2_namespace}/eef_poses")
    print(f"[INFO]: End-effector wrenches published to: {args_cli.ros2_namespace}/eef_wrenches")
    print(f"[INFO]: End-effector velocities published to: {args_cli.ros2_namespace}/eef_velocities")

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

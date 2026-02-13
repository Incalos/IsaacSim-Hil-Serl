import argparse
import os
import sys

# Calculate the project root path relative to this file and inject it into the system path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define command-line arguments for environment scaling, task selection, and ROS2 scoping
parser = argparse.ArgumentParser(description="OmniGraph robot control for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="SO101-PickOranges", help="Name of the task.")
parser.add_argument("--ros2_namespace", type=str, default="so101", help="ROS2 namespace for topics.")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import carb
import carb.input
from isaaclab_tasks.utils import parse_env_cfg
import omni.appwindow
from robot_infra.isaacsim_venvs.tasks import *
from robot_infra.isaacsim_venvs.tasks.so101_pick_oranges.omni_graph.robot_controller import (
    SO101_OmniGraph_Controller,
)


def main():
    # Load task configuration and initialize the Gym-compatible simulation environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.recorders = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    # Instantiate the OmniGraph controller to bridge Isaac Sim with ROS2 and cuRobo IK
    SO101_OmniGraph_Controller(env, ros2_namespace=args_cli.ros2_namespace)

    def on_keyboard_event(event, *args, **kwargs):
        # Filter for key press events only
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        # Manually trigger domain randomization (DR) logic when the 'R' key is pressed
        if event.input == carb.input.KeyboardInput.R:
            base_env = env.unwrapped
            all_env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
            idx = 0
            while True:
                term_name = f"domain_randomize_{idx}"
                term_cfg = getattr(env_cfg.events, term_name, None)
                if term_cfg is None:
                    break
                term_cfg.func(base_env, all_env_ids, **term_cfg.params)
                idx += 1
        simulation_app.update()
        return True

    # Register the keyboard callback with the application's input interface
    app_window = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    # Print initialization status and active ROS2 topic mappings for user reference
    print("[INFO]: OmniGraph joint control started with ROS2 bridge and cuRobo IK.")
    print("[INFO]: Key R = domain randomization.")
    print(f"[INFO]: ROS2 namespace: {args_cli.ros2_namespace}")
    print(f"[INFO]: Joint commands subscribed from: {args_cli.ros2_namespace}/joint_commands")
    print(f"[INFO]: End-effector commands subscribed from: {args_cli.ros2_namespace}/eef_commands")
    print(f"[INFO]: Wrist camera published to: {args_cli.ros2_namespace}/wrist_camera/rgb")
    print(f"[INFO]: Front camera published to: {args_cli.ros2_namespace}/front_camera/rgb")
    print(f"[INFO]: Side camera published to: {args_cli.ros2_namespace}/side_camera/rgb")
    print(f"[INFO]: Joint states published on: {args_cli.ros2_namespace}/joint_states")
    print(f"[INFO]: Joint forces published on: {args_cli.ros2_namespace}/joint_forces")
    print(f"[INFO]: Joint torques published on: {args_cli.ros2_namespace}/joint_torques")
    print(f"[INFO]: End-effector states published to: {args_cli.ros2_namespace}/eef_poses")
    print(f"[INFO]: End-effector wrenches published to: {args_cli.ros2_namespace}/eef_wrenches")
    print(f"[INFO]: End-effector velocities published to: {args_cli.ros2_namespace}/eef_velocities")
    try:
        # Keep the simulation running until the window is closed
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

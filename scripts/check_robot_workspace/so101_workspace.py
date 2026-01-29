"""Determine and visualize the SO101 robot arm task workspace."""

import sys
import os


# Resolve project root so relative imports/assets work when running this script directly.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import multiprocessing
import argparse
from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

# Ensure Isaac Sim uses the spawn start method for multiprocessing.
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


# CLI configuration for workspace exploration.
parser = argparse.ArgumentParser(description="Determine SO101 robot arm workspace for specific tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--recalibrate", action="store_true", help="Recalibrate SO101-Leader or Bi-SO101Leader.")
parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port for the SO101 teleop device.")
parser.add_argument("--task", type=str, default="SO101-PickOranges", help="Name of the Isaac Lab task to launch.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity scaling for teleoperation inputs.")
parser.add_argument("--step_hz", type=int, default=60, help="Target simulation step frequency (Hz).")
parser.add_argument("--quality", action="store_true", help="Enable higher-quality rendering.")
parser.add_argument(
    "--yaml_file",
    type=str,
    default="robot_infra/ros2_ws/src/so101_interfaces/config/so101_params.yaml",
    help="YAML path used to read/write the workspace bounding box.",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "gamepad", "so101leader"],
    help="Device used to control the robot during exploration.",
)

# Integrate Isaac Lab app launcher options and start the app.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import yaml
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from utils.rate_limiter import RateLimiter
from utils.robot_math import isaac_quat_to_scipy_quat, quaternion_to_euler
import gymnasium as gym
from robot_infra.isaacsim_venvs import *
from isaaclab_tasks.utils import parse_env_cfg  # type: ignore[import-not-found]
import isaacsim.util.debug_draw._debug_draw as omni_debug_draw  # type: ignore[import-not-found]
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim  # type: ignore[import-not-found]


def update_quaternion_bounds(current_quat: np.ndarray, min_euler: np.ndarray, max_euler: np.ndarray) -> tuple:
    """Update Euler-space orientation bounds using the current quaternion."""
    current_euler = quaternion_to_euler(current_quat)
    min_euler = np.minimum(min_euler, current_euler)
    max_euler = np.maximum(max_euler, current_euler)
    return min_euler, max_euler


def load_bounding_box(yaml_file: str) -> dict | None:
    """Load an existing workspace bounding box from YAML, if available."""
    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(project_root, yaml_file)

    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
            if data and "bounding_box" in data:
                return data["bounding_box"]
    return None


def save_bounding_box(yaml_file: str, bounding_box: dict):
    """Persist the workspace bounding box to a YAML file."""
    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(project_root, yaml_file)

    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    data["bounding_box"] = bounding_box

    output_dir = os.path.dirname(yaml_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def draw_wireframe_box(draw_interface, min_pos: np.ndarray, max_pos: np.ndarray):
    """Draw a wireframe box between the given min/max corners."""
    x_min, y_min, z_min = min_pos
    x_max, y_max, z_max = max_pos

    vertices = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    start_positions = [vertices[edge[0]] for edge in edges]
    end_positions = [vertices[edge[1]] for edge in edges]
    colors = [[1.0, 0.0, 0.0, 1.0]] * len(edges)
    thicknesses = [3.0] * len(edges)

    draw_interface.draw_lines(start_positions, end_positions, colors, thicknesses)


def main():
    """Run the interactive workspace exploration loop for the SO101 robot arm."""
    # 1) Configure the Isaac Lab environment.
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    # Optionally enable higher-quality rendering (slower but better visuals).
    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # Disable built-in time/success termination so the loop is user-driven.
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    # We only care about workspace exploration here, so disable recorders.
    env_cfg.recorders = None

    # 2) Create the gym environment.
    env = gym.make(task_name, cfg=env_cfg).unwrapped

    # 3) Construct the teleoperation interface based on the selected device.
    if args_cli.teleop_device == "keyboard":
        from leisaac.devices import SO101Keyboard  # type: ignore[import-not-found]

        teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "gamepad":
        from leisaac.devices import SO101Gamepad  # type: ignore[import-not-found]

        teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        from leisaac.devices import SO101Leader  # type: ignore[import-not-found]

        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'gamepad', 'so101leader'."
        )

    # Flag and callback to allow resetting the exploration session via keyboard/gamepad.
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.display_controls()

    # Rate limiter keeps the main loop close to the requested step frequency.
    rate_limiter = RateLimiter(args_cli.step_hz)

    # Acquire debug-draw interface for visualizing the workspace box in the viewer.
    try:
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    except Exception:
        draw_interface = None
        print("[WARNING] Could not acquire debug draw interface. Visualization will be disabled.")

    # Initialize simulation and teleoperation state.
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    # If a bounding box already exists on disk, load it so we can extend it.
    existing_bbox = load_bounding_box(args_cli.yaml_file)

    robot = env.scene["robot"]
    env_id = 0
    ee_frame = env.scene["ee_frame"]

    # Read initial end-effector pose as the starting workspace sample.
    ee_pos = ee_frame.data.target_pos_source[env_id, 0, :].cpu().numpy()
    ee_quat_isaac = ee_frame.data.target_quat_source[env_id, 0, :].cpu().numpy()

    if ee_quat_isaac is None or ee_quat_isaac.shape[0] != 4:
        ee_quat_isaac = np.array([1.0, 0.0, 0.0, 0.0])
        print("[WARNING] Could not find end effector rotation, using identity quaternion")

    ee_quat = isaac_quat_to_scipy_quat(ee_quat_isaac)

    # Initialize translation and orientation bounds from current pose or from YAML.
    if existing_bbox:
        max_translation = existing_bbox.get("max_translation")
        min_translation = existing_bbox.get("min_translation")
        max_rotation = existing_bbox.get("max_rotation")
        min_rotation = existing_bbox.get("min_rotation")

        max_ee_pos = np.array(max_translation if max_translation is not None else ee_pos)
        min_ee_pos = np.array(min_translation if min_translation is not None else ee_pos)

        if max_rotation is not None and min_rotation is not None:
            max_euler = np.array(max_rotation)
            min_euler = np.array(min_rotation)
        else:
            ee_euler = quaternion_to_euler(ee_quat)
            max_euler = ee_euler.copy()
            min_euler = ee_euler.copy()
        print(f"Loaded existing bounding box from {args_cli.yaml_file}")
    else:
        max_ee_pos = ee_pos.copy()
        min_ee_pos = ee_pos.copy()
        ee_euler = quaternion_to_euler(ee_quat)
        max_euler = ee_euler.copy()
        min_euler = ee_euler.copy()

    # Periodically persist the updated bounds to disk.
    save_frequency = 10
    step_count = 0

    # 4) Main simulation loop: drive the robot, update bounds and visualize workspace.
    while simulation_app.is_running():
        with torch.inference_mode():
            # Optionally adapt the gripper effort limit based on the current task/device.
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)

            # Read teleop actions for this step.
            actions = teleop_interface.advance()

            # Query current end-effector pose in the robot base frame.
            ee_pos = ee_frame.data.target_pos_source[env_id, 0, :].cpu().numpy()
            ee_quat_isaac = ee_frame.data.target_quat_source[env_id, 0, :].cpu().numpy()

            if ee_quat_isaac is None or ee_quat_isaac.shape[0] != 4:
                # Skip this step if we cannot get a valid end-effector orientation.
                continue

            ee_quat = isaac_quat_to_scipy_quat(ee_quat_isaac)

            # Update translation bounds in base frame.
            max_ee_pos = np.maximum(max_ee_pos, ee_pos)
            min_ee_pos = np.minimum(min_ee_pos, ee_pos)

            # Update orientation bounds in Euler space.
            min_euler, max_euler = update_quaternion_bounds(ee_quat, min_euler, max_euler)

            # Visualize the workspace box in world coordinates, if debug draw is available.
            if draw_interface is not None:
                draw_interface.clear_lines()

                base_pos_w = robot.data.root_pos_w[env_id, :].cpu().numpy()
                base_quat_w_isaac = robot.data.root_quat_w[env_id, :].cpu().numpy()
                base_quat_w = isaac_quat_to_scipy_quat(base_quat_w_isaac)

                base_rotation = Rotation.from_quat(base_quat_w)

                x_min, y_min, z_min = min_ee_pos
                x_max, y_max, z_max = max_ee_pos
                vertices_base = np.array(
                    [
                        [x_min, y_min, z_min],
                        [x_max, y_min, z_min],
                        [x_max, y_max, z_min],
                        [x_min, y_max, z_min],
                        [x_min, y_min, z_max],
                        [x_max, y_min, z_max],
                        [x_max, y_max, z_max],
                        [x_min, y_max, z_max],
                    ]
                )

                # Transform the box from base frame into world frame.
                vertices_world = base_rotation.apply(vertices_base) + base_pos_w
                min_ee_pos_w = vertices_world.min(axis=0)
                max_ee_pos_w = vertices_world.max(axis=0)

                draw_wireframe_box(draw_interface, min_ee_pos_w, max_ee_pos_w)

            step_count += 1
            if step_count >= save_frequency:
                bounding_box = {
                    "min_translation": min_ee_pos.tolist(),
                    "max_translation": max_ee_pos.tolist(),
                    "min_rotation": min_euler.tolist(),
                    "max_rotation": max_euler.tolist(),
                }
                save_bounding_box(args_cli.yaml_file, bounding_box)
                print(f"Max ee position: {np.round(max_ee_pos, 4).tolist()}")
                print(f"Min ee position: {np.round(min_ee_pos, 4).tolist()}")
                print(f"Max ee rotation (Euler [yaw,pitch,roll] deg): {np.round(max_euler, 4).tolist()}")
                print(f"Min ee rotation (Euler [yaw,pitch,roll] deg): {np.round(min_euler, 4).tolist()}")
                print("==" * 20)
                step_count = 0

            # Allow the user to reset the environment and start a fresh exploration trace.
            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

            # Step the environment using current actions (or just render if no actions).
            if actions is None:
                env.render()
            else:
                env.step(actions)

            # Sleep as needed to honor the requested control rate.
            if rate_limiter:
                rate_limiter.sleep(env)

    # 5) Save final bounds when the app exits.
    bounding_box = {
        "min_translation": min_ee_pos.tolist(),
        "max_translation": max_ee_pos.tolist(),
        "min_rotation": min_euler.tolist(),
        "max_rotation": max_euler.tolist(),
    }
    save_bounding_box(args_cli.yaml_file, bounding_box)
    print(f"Final bounding box saved to {args_cli.yaml_file}")
    print(f"Max ee position: {np.round(max_ee_pos, 4).tolist()}")
    print(f"Min ee position: {np.round(min_ee_pos, 4).tolist()}")
    print(f"Max ee rotation (Euler [yaw,pitch,roll] deg): {np.round(max_euler, 4).tolist()}")
    print(f"Min ee rotation (Euler [yaw,pitch,roll] deg): {np.round(min_euler, 4).tolist()}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

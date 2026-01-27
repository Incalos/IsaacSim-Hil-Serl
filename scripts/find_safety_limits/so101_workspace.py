"""Script to determine and visualize the workspace of SO101 robot arm for specific tasks.

This script tracks the end effector position and orientation during manual teleoperation
to determine the reachable workspace boundaries for task-specific applications.
The workspace is defined as a bounding box containing all reachable positions and orientations.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

parser = argparse.ArgumentParser(description="Determine SO101 robot arm workspace for specific tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port", type=str, default="/dev/ttyACM0", help="Port for the teleop device:so101leader, default is /dev/ttyACM0"
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument("--task", type=str, default="SO101-PickOranges", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")
parser.add_argument(
    "--yaml_file",
    type=str,
    default="ros2_workspace/src/so101_interfaces/config/so101_params.yaml",
    help="Path to YAML file.",
)
parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import yaml
import time
import numpy as np
import torch
import gymnasium as gym
from robot_envs.isaacsim_envs import *
import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


def load_bounding_box(yaml_file: str) -> dict | None:
    """Load existing workspace bounding box from YAML configuration file.

    The bounding box represents previously determined workspace limits for the task.
    This allows resuming workspace exploration from a saved state.
    """
    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(project_root, yaml_file)

    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
            if data and "bounding_box" in data:
                return data["bounding_box"]
    return None


def save_bounding_box(yaml_file: str, bounding_box: dict):
    """Save workspace bounding box to YAML configuration file.

    The bounding box contains min/max translation and rotation limits
    that define the reachable workspace for the specific task.
    """
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


def draw_wireframe_box(
    draw_interface, min_pos: np.ndarray, max_pos: np.ndarray, color: list = [1.0, 0.0, 0.0, 1.0], thickness: float = 3.0
):
    """Visualize workspace boundaries as a wireframe bounding box.

    The box represents the spatial extent of the robot's reachable workspace
    for the current task, helping operators understand workspace limits.
    """
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
    colors = [color] * len(edges)
    thicknesses = [thickness] * len(edges)

    draw_interface.draw_lines(start_positions, end_positions, colors, thicknesses)


class RateLimiter:
    """Enforce consistent simulation rate for workspace exploration."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Maintain target simulation frequency."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
    """Main workspace determination workflow for SO101 robot arm.

    This function sets up the simulation environment, enables teleoperation,
    and continuously tracks end effector pose to determine task-specific workspace.
    The workspace is defined by the minimum and maximum reachable positions
    and orientations during manual exploration.
    """
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    if "BiArm" in task_name:
        assert args_cli.teleop_device == "bi-so101leader", "only support bi-so101leader for bi-arm task"
    if "LeKiwi" in task_name:
        assert args_cli.teleop_device in [
            "lekiwi-leader",
            "lekiwi-keyboard",
            "lekiwi-gamepad",
        ], "only support lekiwi-leader, lekiwi-keyboard, lekiwi-gamepad for lekiwi task"

    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env_cfg.recorders = None

    env = gym.make(task_name, cfg=env_cfg).unwrapped

    if args_cli.teleop_device == "keyboard":
        from leisaac.devices import SO101Keyboard

        teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "gamepad":
        from leisaac.devices import SO101Gamepad

        teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        from leisaac.devices import SO101Leader

        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "bi-so101leader":
        from leisaac.devices import BiSO101Leader

        teleop_interface = BiSO101Leader(
            env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=args_cli.recalibrate
        )
    elif args_cli.teleop_device == "lekiwi-keyboard":
        from leisaac.devices import LeKiwiKeyboard

        teleop_interface = LeKiwiKeyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "lekiwi-leader":
        from leisaac.devices import LeKiwiLeader

        teleop_interface = LeKiwiLeader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "lekiwi-gamepad":
        from leisaac.devices import LeKiwiGamepad

        teleop_interface = LeKiwiGamepad(env, sensitivity=args_cli.sensitivity)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'gamepad', 'so101leader',"
            " 'bi-so101leader', 'lekiwi-keyboard', 'lekiwi-leader', 'lekiwi-gamepad'."
        )

    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.display_controls()

    rate_limiter = RateLimiter(args_cli.step_hz)

    try:
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    except Exception:
        draw_interface = None
        print("[WARNING] Could not acquire debug draw interface. Visualization will be disabled.")

    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    existing_bbox = load_bounding_box(args_cli.yaml_file)

    robot = env.scene["robot"]
    env_id = 0

    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[env_id, 0, :].cpu().numpy()
    ee_quat = ee_frame.data.target_quat_w[env_id, 0, :].cpu().numpy()

    if ee_quat is None:
        ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
        print("[WARNING] Could not find end effector rotation, using identity quaternion")

    if existing_bbox:
        max_translation = existing_bbox.get("max_translation")
        min_translation = existing_bbox.get("min_translation")
        max_rotation = existing_bbox.get("max_rotation")
        min_rotation = existing_bbox.get("min_rotation")

        max_ee_pos = np.array(max_translation if max_translation is not None else ee_pos)
        min_ee_pos = np.array(min_translation if min_translation is not None else ee_pos)
        max_ee_quat = np.array(max_rotation if max_rotation is not None else ee_quat)
        min_ee_quat = np.array(min_rotation if min_rotation is not None else ee_quat)
        print(f"Loaded existing bounding box from {args_cli.yaml_file}")
    else:
        max_ee_pos = ee_pos.copy()
        min_ee_pos = ee_pos.copy()
        max_ee_quat = ee_quat.copy()
        min_ee_quat = ee_quat.copy()

    save_frequency = 60
    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)

            actions = teleop_interface.advance()

            ee_pos = ee_frame.data.target_pos_w[env_id, 0, :].cpu().numpy()

            ee_quat = ee_frame.data.target_quat_w[env_id, 0, :].cpu().numpy()

            if ee_quat is None or ee_quat.shape[0] != 4:
                continue

            max_ee_pos = np.maximum(max_ee_pos, ee_pos)
            min_ee_pos = np.minimum(min_ee_pos, ee_pos)
            max_ee_quat = np.maximum(max_ee_quat, ee_quat)
            min_ee_quat = np.minimum(min_ee_quat, ee_quat)

            if draw_interface is not None:
                draw_interface.clear_lines()
                draw_wireframe_box(draw_interface, min_ee_pos, max_ee_pos, color=[1.0, 0.0, 0.0, 1.0], thickness=3.0)

            step_count += 1
            if step_count >= save_frequency:
                bounding_box = {
                    "min_translation": min_ee_pos.tolist(),
                    "max_translation": max_ee_pos.tolist(),
                    "min_rotation": min_ee_quat.tolist(),
                    "max_rotation": max_ee_quat.tolist(),
                }
                save_bounding_box(args_cli.yaml_file, bounding_box)
                print(f"Max ee position: {np.round(max_ee_pos, 4).tolist()}")
                print(f"Min ee position: {np.round(min_ee_pos, 4).tolist()}")
                print(f"Max ee rotation (quaternion [x,y,z,w]): {np.round(max_ee_quat, 4).tolist()}")
                print(f"Min ee rotation (quaternion [x,y,z,w]): {np.round(min_ee_quat, 4).tolist()}")
                print("==" * 20)
                step_count = 0

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

            if actions is None:
                env.render()
            else:
                env.step(actions)

            if rate_limiter:
                rate_limiter.sleep(env)

    bounding_box = {
        "min_translation": min_ee_pos.tolist(),
        "max_translation": max_ee_pos.tolist(),
        "min_rotation": min_ee_quat.tolist(),
        "max_rotation": max_ee_quat.tolist(),
    }
    save_bounding_box(args_cli.yaml_file, bounding_box)
    print(f"Final bounding box saved to {args_cli.yaml_file}")
    print(f"Max ee position: {np.round(max_ee_pos, 4).tolist()}")
    print(f"Min ee position: {np.round(min_ee_pos, 4).tolist()}")
    print(f"Max ee rotation (quaternion [x,y,z,w]): {np.round(max_ee_quat, 4).tolist()}")
    print(f"Min ee rotation (quaternion [x,y,z,w]): {np.round(min_ee_quat, 4).tolist()}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

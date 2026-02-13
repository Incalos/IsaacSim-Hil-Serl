import copy
from pathlib import Path
import struct
import time
from typing import Dict, Sequence
import cv2
import gymnasium as gym
import numpy as np
from pynput import keyboard
import requests
from pytransform3d.urdf import UrdfTransformManager
from scipy.spatial.transform import Rotation


class DefaultEnvConfig:
    # Hardware communication and asset paths
    SERVER_URL: str = "http://127.0.0.1:5000"
    URDF_PATH: str = "isaacsim_venvs/assets/robots/so101_follower.urdf"
    CAMERA_NAMES: list[str] = ["wrist_camera", "front_camera", "side_camera"]
    # ROI (Region of Interest) cropping to focus on specific workspace areas
    IMAGE_CROP: dict[str, callable] = {
        "front_camera": lambda img: img,
        "wrist_camera": lambda img: img,
        "side_camera": lambda img: img,
    }
    # Scaling factors for translation (m), rotation (rad), and gripper normalized movement
    ACTION_SCALE = (0.05, 0.2, 1)
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    MAX_EPISODE_LENGTH: int = 200


class SO101Env(gym.Env):
    def __init__(self, fake_env=False, hz=10, config=DefaultEnvConfig()):
        self.hz = hz
        self.config = config
        self.url = config.SERVER_URL
        self.action_scale = config.ACTION_SCALE
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.camera_names = config.CAMERA_NAMES
        self.indexes = [0, 3, 4]
        self.xyz_bounding_box = self.rpy_bounding_box = None
        # Initialize URDF Manager for local Forward Kinematics calculations
        urdf_path = Path(config.URDF_PATH)
        if not urdf_path.is_absolute():
            _root = Path(__file__).resolve().parent.parent
            urdf_path = _root / urdf_path
        self.tm = UrdfTransformManager()
        with open(urdf_path, "r") as f:
            urdf_text = f.read()
        self.tm.load_urdf(urdf_text)
        # Pull robot-specific limits and configurations from the remote server
        self._load_yaml_config()
        self._update_currpos()
        # Define normalized Action Space
        self.action_space = gym.spaces.Box(np.ones((6,), dtype=np.float32) * -1, np.ones((6,), dtype=np.float32))
        # Define multi-modal Observation Space
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "q": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "dq": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) for key in self.camera_names}
                ),
            }
        )
        if fake_env:
            return
        self.curr_path_length = 0
        self.terminate = False

        # Asynchronous listener for safe manual shutdown via 'Esc'
        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        print("Initialized SO101")

    def _update_currpos(self):
        # Sync all robot state variables with the remote server via a single POST request
        ps = requests.post(self.url + "/get_state").json()
        self.curr_joint_positions = np.array(ps["joint_positions"])
        self.curr_joint_velocities = np.array(ps["joint_velocities"])
        self.curr_eef_poses_quat = np.array(ps["eef_poses_quat"])
        self.curr_eef_poses_euler = np.array(ps["eef_poses_euler"])
        self.curr_eef_forces = np.array(ps["eef_forces"])
        self.curr_eef_torques = np.array(ps["eef_torques"])
        self.curr_eef_velocities = np.array(ps["eef_velocities"])

    def _load_yaml_config(self):
        # Fetch operational boundaries and hardware limits from the server registry
        ps = requests.post(self.url + "/get_config").json()
        self.joint_names = ps["joint_names"]
        self.reset_pose = np.array(ps["reset_joint_positions"])
        if (
            ps["min_translation"] is not None
            and ps["max_translation"] is not None
            and ps["min_rotation"] is not None
            and ps["max_rotation"] is not None
        ):
            self.xyz_bounding_box = gym.spaces.Box(
                np.array(ps["min_translation"]), np.array(ps["max_translation"]), dtype=np.float64
            )
            self.rpy_bounding_box = gym.spaces.Box(
                np.array(ps["min_rotation"]), np.array(ps["max_rotation"]), dtype=np.float64
            )
        self.shoulder_pan_limits = ps["shoulder_pan_limits"]
        self.shoulder_lift_limits = ps["shoulder_lift_limits"]
        self.elbow_flex_limits = ps["elbow_flex_limits"]
        self.wrist_flex_limits = ps["wrist_flex_limits"]
        self.wrist_roll_limits = ps["wrist_roll_limits"]
        self.gripper_limits = ps["gripper_limits"]

    def joints_to_eef_fk(self, joint_positions: Sequence[float]) -> np.ndarray:
        # Use the URDF manager to compute the Cartesian pose of the gripper given joint angles
        gripper_position = joint_positions[-1]
        for name, pos in zip(self.joint_names, joint_positions[:-1]):
            self.tm.set_joint(name, pos)
        T = self.tm.get_transform("so101_new_calib_gripper", "so101_new_calib")
        position = T[:3, 3]
        quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        if quat_xyzw[3] < 0:
            quat_xyzw = -quat_xyzw
        return np.concatenate([position, quat_xyzw, [gripper_position]])

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        # Hardware protection: Enforce hard limits on Cartesian position and Euler orientation
        if self.xyz_bounding_box is not None and self.rpy_bounding_box is not None:
            pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
            euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
            sign = np.sign(euler[2])
            euler[2] = sign * (np.clip(np.abs(euler[2]), self.rpy_bounding_box.low[2], self.rpy_bounding_box.high[2]))
            euler[:2] = np.clip(euler[:2], self.rpy_bounding_box.low[:2], self.rpy_bounding_box.high[:2])
            pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
            return pose
        else:
            return pose

    def get_im(self, quality: int | None = None) -> Dict[str, np.ndarray]:
        # Batch fetch images; Uses custom binary protocol (Header + JPEG segments) for efficiency
        try:
            url = f"{self.url}/get_images"
            resp = requests.post(url, timeout=2, params={"quality": quality} if quality is not None else None)
            if resp.status_code == 200:
                names = [n.strip() for n in resp.headers.get("X-Camera-Names", "").split(",") if n.strip()]
                body, images, offset = resp.content, {}, 0
                for name in names:
                    if offset + 4 > len(body):
                        break
                    (length,) = struct.unpack(">I", body[offset : offset + 4])
                    offset += 4
                    rgb = cv2.imdecode(np.frombuffer(body[offset : offset + length], np.uint8), cv2.IMREAD_COLOR)
                    offset += length
                    if rgb is not None:
                        # Process: Crop to ROI -> Resize to model input -> Convert BGR to RGB
                        cropped = self.config.IMAGE_CROP[name](rgb) if name in self.config.IMAGE_CROP else rgb
                        images[name] = cv2.resize(cropped, self.observation_space["images"][name].shape[:2][::-1])
                return images
        except Exception as e:
            print(f"Error fetching cameras: {e}")
            input("Press enter to retry...")
            return self.get_im(quality)

    def _send_eef_command(self, pos: np.ndarray, gripper_state: float):
        # Command the robot via End-Effector Cartesian target
        data = {"eef_pose": np.array(pos).astype(np.float32).tolist(), "gripper_state": gripper_state}
        requests.post(self.url + "/move_eef", json=data)

    def _send_joint_command(self, pos: np.ndarray):
        # Command the robot via raw Joint Angle target
        data = {"joint_pose": np.array(pos).astype(np.float32).tolist()}
        requests.post(self.url + "/move_joints", json=data)

    def _get_obs(self) -> dict:
        # Aggregate proprioceptive data and visual streams into a single observation dict
        return copy.deepcopy(
            dict(
                images=self.get_im(),
                state={
                    "tcp_pose": self.curr_eef_poses_quat,
                    "tcp_vel": self.curr_eef_velocities,
                    "tcp_force": self.curr_eef_forces,
                    "tcp_torque": self.curr_eef_torques,
                    "q": self.curr_joint_positions,
                    "dq": self.curr_joint_velocities,
                },
            )
        )

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_joint_positions = self.curr_joint_positions.copy()
        next_joint_positions[self.indexes] += action[self.indexes] * self.action_scale[1]
        next_joint_positions[-1] += action[-1] * self.action_scale[2]
        self.nextpos = self.joints_to_eef_fk(next_joint_positions)
        self.nextpos[1] += action[1] * self.action_scale[0]
        self.nextpos[2] += action[2] * self.action_scale[0]
        self._send_eef_command(self.clip_safety_box(self.nextpos[:7]), self.nextpos[-1])
        # Control Loop Timing: Maintain 'hz' frequency by sleeping for the remaining budget
        self.curr_path_length += 1
        time.sleep(max(0, (1.0 / self.hz) - (time.time() - start_time)))
        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, ob) -> bool:
        return False  # Base implementation: Sparse reward defined in task-specific wrappers

    def go_to_reset(self):
        # Reset logic with optional Gaussian noise for Domain Randomization
        self._update_currpos()
        if self.randomreset:
            reset_pose = self.joints_to_eef_fk(self.reset_pose.copy())
            reset_pose[:2] += np.random.uniform(-self.random_xy_range, self.random_xy_range, (2,))
            euler_random = Rotation.from_quat(reset_pose[3:7]).as_euler("xyz")
            euler_random[-1] += np.random.uniform(-self.random_rz_range, self.random_rz_range)
            reset_pose[3:7] = Rotation.from_euler("xyz", euler_random).as_quat()
            self._send_eef_command(reset_pose[:7], reset_pose[-1])
        else:
            requests.post(self.url + "/reset_robot")

    def reset(self, **kwargs):
        self.go_to_reset()
        time.sleep(1)  # Allow settling time for the robot hardware
        self.curr_path_length, self.terminate = 0, False
        self._update_currpos()
        return self._get_obs(), {"succeed": False}

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()

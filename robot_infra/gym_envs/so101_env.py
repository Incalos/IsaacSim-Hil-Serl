import copy
import sys
import struct
import time
import yaml
from typing import Dict
import cv2
import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation

try:
    from pynput import keyboard
except ImportError:

    class MockKeyboard:

        class Listener:

            @staticmethod
            def on_press(*args, **kwargs):
                pass

            @staticmethod
            def on_release(*args, **kwargs):
                pass

            @staticmethod
            def start(*args, **kwargs):
                return MockKeyboard.Listener()

            @staticmethod
            def join(*args, **kwargs):
                pass

    keyboard = MockKeyboard()
    print("Warning: pynput could not be imported, mock class has been used instead", file=sys.stderr)


class DefaultEnvConfig:
    SERVER_URL: str = "http://127.0.0.1:5000"
    CAMERA_NAMES: list[str] = ["wrist_camera", "front_camera", "side_camera"]
    MAX_EPISODE_LENGTH: int = 200
    IMAGE_CROP: dict[str, callable]
    ACTION_SCALE: tuple[float]
    ROBOT_CONFIG: str


class SO101Env(gym.Env):

    def __init__(self, fake_env=False, hz=10, config=DefaultEnvConfig(), image_size=(128, 128)):
        # Initialize core configuration parameters
        self.hz = hz
        self.config = config
        self.url = config.SERVER_URL
        self.action_scale = config.ACTION_SCALE
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.camera_names = config.CAMERA_NAMES
        self.xyz_bounding_box = self.rpy_bounding_box = None

        # Define normalized action space (7 DOF: 6 for EEF pose, 1 for gripper)
        self.action_space = gym.spaces.Box(
            low=np.full(7, -1.0, dtype=np.float32),
            high=np.full(7, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # Define multi-modal observation space (robot state + multi-camera RGB images)
        self.observation_space = gym.spaces.Dict({
            "state":
                gym.spaces.Dict({
                    "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    "q": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    "dq": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                }),
            "images":
                gym.spaces.Dict({key: gym.spaces.Box(0, 255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8) for key in self.camera_names}),
        })

        # Load robot configuration parameters from YAML file
        self._load_yaml_config(config.ROBOT_CONFIG)

        # Skip real environment initialization if fake_env flag is enabled
        if fake_env:
            return

        # Synchronize initial robot state from remote server
        self._update_currpos()
        self.curr_path_length = 0
        self.terminate = False

        # Setup emergency shutdown via Esc key press
        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        print("Initialized SO101")

    def _update_currpos(self):
        # Synchronize robot state with remote server via HTTP POST request
        ps = requests.post(self.url + "/get_state").json()
        self.curr_joint_positions = np.array(ps["joint_positions"])
        self.curr_joint_velocities = np.array(ps["joint_velocities"])
        self.curr_eef_poses_quat = np.array(ps["eef_poses_quat"])
        self.curr_eef_poses_euler = np.array(ps["eef_poses_euler"])
        self.curr_eef_forces = np.array(ps["eef_forces"])
        self.curr_eef_torques = np.array(ps["eef_torques"])
        self.curr_eef_velocities = np.array(ps["eef_velocities"])

    def _load_yaml_config(self, path):
        # Load robot parameters from YAML configuration file
        with open(path, "r") as f:
            params = yaml.safe_load(f)
            self.joint_names = params["joint_names"]
            self.reset_pose = params.get("reset_joint_positions", [])

            # Initialize Cartesian and rotation bounding boxes if configured
            if (params["bounding_box"]["min_translation"] is not None and params["bounding_box"]["max_translation"] is not None and params["bounding_box"]["min_rotation"] is not None and
                    params["bounding_box"]["max_rotation"] is not None):
                self.xyz_bounding_box = gym.spaces.Box(
                    np.array(params["bounding_box"]["min_translation"]),
                    np.array(params["bounding_box"]["max_translation"]),
                    dtype=np.float64,
                )
                self.rpy_bounding_box = gym.spaces.Box(
                    np.array(params["bounding_box"]["min_rotation"]),
                    np.array(params["bounding_box"]["max_rotation"]),
                    dtype=np.float64,
                )

            # Load joint limits for each robot joint
            self.shoulder_pan_limits = params["joint_limits"]["shoulder_pan"]
            self.shoulder_lift_limits = params["joint_limits"]["shoulder_lift"]
            self.elbow_flex_limits = params["joint_limits"]["elbow_flex"]
            self.wrist_flex_limits = params["joint_limits"]["wrist_flex"]
            self.wrist_roll_limits = params["joint_limits"]["wrist_roll"]
            self.gripper_limits = params["joint_limits"]["gripper"]
            self.urdf_path = params["urdf_path"]

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        # Enforce safety limits for Cartesian position and rotation
        if self.xyz_bounding_box is not None and self.rpy_bounding_box is not None:
            # Clip translation (XYZ) to defined bounding box
            pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)

            # Convert quaternion to Euler angles, apply rotation limits, convert back to quaternion
            euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
            sign = np.sign(euler[2])
            euler[2] = sign * (np.clip(np.abs(euler[2]), self.rpy_bounding_box.low[2], self.rpy_bounding_box.high[2]))
            euler[:2] = np.clip(euler[:2], self.rpy_bounding_box.low[:2], self.rpy_bounding_box.high[:2])
            pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
            return pose
        else:
            return pose

    def get_im(self, quality: int | None = None) -> Dict[str, np.ndarray]:
        # Fetch camera images from remote server using binary protocol
        try:
            url = f"{self.url}/get_images"
            resp = requests.post(url, timeout=2, params={"quality": quality} if quality is not None else None)
            if resp.status_code == 200:
                # Parse camera names from response headers
                names = [n.strip() for n in resp.headers.get("X-Camera-Names", "").split(",") if n.strip()]
                body, images, offset = resp.content, {}, 0

                # Parse binary payload (length prefix + JPEG data)
                for name in names:
                    if offset + 4 > len(body):
                        break
                    (length,) = struct.unpack(">I", body[offset:offset + 4])
                    offset += 4
                    rgb = cv2.imdecode(np.frombuffer(body[offset:offset + length], np.uint8), cv2.IMREAD_COLOR)
                    offset += length

                    # Process image: crop -> resize -> BGR to RGB conversion
                    if rgb is not None:
                        cropped = self.config.IMAGE_CROP[name](rgb) if name in self.config.IMAGE_CROP else rgb
                        resized = cv2.resize(cropped, self.observation_space["images"][name].shape[:2][::-1])
                        images[name] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                return images
        except Exception as e:
            print(f"Error fetching cameras: {e}")
            input("Press enter to retry...")
            return self.get_im(quality)

    def _send_eef_command(self, pos: np.ndarray, gripper_state: float):
        # Send end-effector Cartesian pose command to remote server
        data = {"eef_pose": np.array(pos).astype(np.float32).tolist(), "gripper_state": gripper_state}
        requests.post(self.url + "/move_eef", json=data)

    def _send_joint_command(self, pos: np.ndarray):
        # Send joint position command to remote server
        data = {"joint_pose": np.array(pos).astype(np.float32).tolist()}
        requests.post(self.url + "/move_joints", json=data)

    def _get_obs(self) -> dict:
        # Aggregate observation data (robot state + camera images)
        return copy.deepcopy(
            dict(
                images=self.get_im(),
                state={
                    "tcp_pose": self.curr_eef_poses_euler,
                    "tcp_vel": self.curr_eef_velocities,
                    "tcp_force": self.curr_eef_forces,
                    "tcp_torque": self.curr_eef_torques,
                    "q": self.curr_joint_positions,
                    "dq": self.curr_joint_velocities,
                },
            ))

    def step(self, action: np.ndarray) -> tuple:
        # Execute one step of the environment
        start_time = time.time()

        # Ensure action is within valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Calculate next end-effector pose based on action and scaling factors
        next_eef_pos = self.curr_eef_poses_quat.copy()
        next_eef_pos[:3] = next_eef_pos[:3] + action[:3] * self.action_scale[0]
        next_eef_pos[3:] = (Rotation.from_rotvec(action[3:6] * self.action_scale[1]) * Rotation.from_quat(self.curr_eef_poses_quat[3:])).as_quat()

        # Calculate and clip gripper action to physical limits
        gripper_action = float(self.curr_joint_positions[-1] + action[6] * self.action_scale[2])
        gripper_action = np.clip(gripper_action, self.gripper_limits[0], self.gripper_limits[1])
        next_eef_poses = self.clip_safety_box(next_eef_pos)

        # Send control command to robot
        self._send_eef_command(next_eef_poses, gripper_action)

        # Increment episode step counter
        self.curr_path_length += 1

        # Maintain control frequency by sleeping if necessary
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        # Update robot state and get current observation
        self._update_currpos()
        ob = self._get_obs()

        # Calculate reward and check termination conditions
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate

        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, ob) -> bool:
        # Base reward function (to be overridden by task-specific implementations)
        return False

    def go_to_reset(self):
        # Reset robot to initial pose via server request
        self._update_currpos()
        requests.post(self.url + "/reset_robot")

    def reset(self, **kwargs):
        # Reset environment to initial state
        self.go_to_reset()
        time.sleep(3.5)
        self.curr_path_length, self.terminate = 0, False
        self._update_currpos()
        return self._get_obs(), {"succeed": False}

    def close(self):
        # Cleanup resources (stop keyboard listener)
        if hasattr(self, "listener"):
            self.listener.stop()
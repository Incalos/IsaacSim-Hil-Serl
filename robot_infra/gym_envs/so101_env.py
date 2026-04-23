import copy
import sys
import struct
import time
import yaml
import cv2
import gymnasium
import numpy as np
import requests
from typing import Dict, Optional, Any, Tuple, Callable
from scipy.spatial.transform import Rotation
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from pynput import keyboard
except ImportError:

    class MockKeyboard:
        class Listener:
            @staticmethod
            def on_press(*args, **kwargs) -> None:
                pass

            @staticmethod
            def on_release(*args, **kwargs) -> None:
                pass

            @staticmethod
            def start(*args, **kwargs) -> "MockKeyboard.Listener":
                return MockKeyboard.Listener()

            @staticmethod
            def join(*args, **kwargs) -> None:
                pass

    keyboard = MockKeyboard()
    print("Warning: pynput could not be imported, mock class has been used instead", file=sys.stderr)


def build_image_crop_functions(crop_config: dict) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Create image cropping functions for each camera from configuration.

    Args:
        crop_config: Dictionary mapping camera names to crop regions with
                     h_start/h_end/w_start/w_end keys

    Returns:
        Dictionary mapping camera names to crop functions that take an image
        and return the cropped region
    """
    crop_fns = {}
    for cam, region in crop_config.items():
        h0 = region["h_start"]
        h1 = region["h_end"]
        w0 = region["w_start"]
        w1 = region["w_end"]

        def crop_func(img: np.ndarray, h0=h0, h1=h1, w0=w0, w1=w1) -> np.ndarray:
            return img[h0:h1, w0:w1, :]

        crop_fns[cam] = crop_func
    return crop_fns


class SO101Env(gymnasium.Env):
    """Gymnasium environment for SO101 robot control via server communication.

    Implements standard gymnasium.Env interface for robot control with:
    - Joint/end-effector position control
    - Camera image observation collection
    - Safety bounding box constraints
    - Keyboard interrupt handling (ESC to terminate)

    Args:
        fake_env: If True, skip server initialization (for testing)
        robot_params: Dictionary with robot parameters
    """

    def __init__(self, fake_env: bool = False, robot_params: dict = None) -> None:
        if robot_params is not None:
            self._load_robot_params(robot_params)
        else:
            raise ValueError("robot_params must be provided when initializing SO101Env")

        self.action_space = gymnasium.spaces.Box(
            low=np.full(7, -1.0, dtype=np.float32),
            high=np.full(7, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gymnasium.spaces.Dict(
            {
                "state": gymnasium.spaces.Dict(
                    {
                        "tcp_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gymnasium.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_force": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "q": gymnasium.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "dq": gymnasium.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    }
                ),
                "images": gymnasium.spaces.Dict(
                    {
                        key: gymnasium.spaces.Box(
                            0, 255, shape=(self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                        )
                        for key in self.camera_names
                    }
                ),
            }
        )

        if fake_env:
            return

        self._init_request_session()
        self._update_currpos()
        self.curr_path_length = 0
        self.terminate = False

        def on_press(key) -> None:
            if key == keyboard.Key.esc:
                self.terminate = True

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        print("Initialized SO101")

    def _load_robot_params(self, robot_params: dict) -> None:
        """Load robot configuration parameters.

        Args:
            robot_params: Dictionary containing robot configuration parameters
        """
        self.joint_names = robot_params["joint_names"]
        self.reset_pose = robot_params.get("reset_joint_positions", [])

        if (
            robot_params["bounding_box"]["min_translation"] is not None
            and robot_params["bounding_box"]["max_translation"] is not None
            and robot_params["bounding_box"]["min_rotation"] is not None
            and robot_params["bounding_box"]["max_rotation"] is not None
        ):
            self.xyz_bounding_box = gymnasium.spaces.Box(
                np.array(robot_params["bounding_box"]["min_translation"]),
                np.array(robot_params["bounding_box"]["max_translation"]),
                dtype=np.float64,
            )
            self.rpy_bounding_box = gymnasium.spaces.Box(
                np.array(robot_params["bounding_box"]["min_rotation"]),
                np.array(robot_params["bounding_box"]["max_rotation"]),
                dtype=np.float64,
            )

        self.shoulder_pan_limits = robot_params["joint_limits"]["shoulder_pan"]
        self.shoulder_lift_limits = robot_params["joint_limits"]["shoulder_lift"]
        self.elbow_flex_limits = robot_params["joint_limits"]["elbow_flex"]
        self.wrist_flex_limits = robot_params["joint_limits"]["wrist_flex"]
        self.wrist_roll_limits = robot_params["joint_limits"]["wrist_roll"]
        self.gripper_limits = robot_params["joint_limits"]["gripper"]

        self.hz = robot_params["policy_hz"]
        self.flask_server = robot_params["flask_server"]
        self.action_scale = robot_params["action_scale"]
        self.max_episode_length = robot_params["max_episode_length"]
        self.camera_names = robot_params["camera_names"]
        self.randomreset = robot_params["random_reset"]
        self.random_joint_range = robot_params["random_joint_range"]
        self.image_size = robot_params["image_size"]
        self.image_crop = build_image_crop_functions(robot_params["image_crop_regions"])

    def _init_request_session(self) -> None:
        """Initialize requests session with retry strategy for robust server communication.

        Configures HTTP adapter with retry logic for transient errors (429, 500-504)
        and sets timeout values for requests.
        """
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.request_timeout = (5, 10)

    def _server_request(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """Send POST request to robot control server with error handling.

        Args:
            endpoint: Server endpoint path (e.g., "/get_state")
            data: JSON-serializable payload to send (default: None)

        Returns:
            JSON response dictionary from server

        Raises:
            requests.exceptions.RequestException: If request fails after retries
        """
        try:
            url = f"{self.flask_server}{endpoint}"
            response = self.session.post(url, json=data, timeout=self.request_timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Server request failed (endpoint: {endpoint}): {e}", file=sys.stderr)
            self._init_request_session()
            raise

    def _update_currpos(self) -> None:
        """Update current robot state from server.

        Fetches and parses joint positions/velocities, end-effector pose,
        velocities, forces and torques from the robot server.
        """
        try:
            ps = self._server_request("/get_state")
            self.curr_joint_positions = np.array(ps["joint_positions"])
            self.curr_joint_velocities = np.array(ps["joint_velocities"])
            self.curr_eef_poses_quat = np.array(ps["eef_poses_quat"])
            self.curr_eef_poses_euler = np.array(ps["eef_poses_euler"])
            self.curr_eef_forces = np.array(ps["eef_forces"])
            self.curr_eef_torques = np.array(ps["eef_torques"])
            self.curr_eef_velocities = np.array(ps["eef_velocities"])
        except Exception as e:
            print(f"Failed to update robot state: {e}", file=sys.stderr)

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip end-effector pose to safety bounding box constraints.

        Constrains both translation (XYZ) and rotation (converted from quaternion
        to RPY for clipping then back to quaternion) to stay within safe limits.

        Args:
            pose: End-effector pose as [x,y,z,qx,qy,qz,qw]

        Returns:
            Clipped pose with all values within safety bounding box limits
        """
        if self.xyz_bounding_box is not None and self.rpy_bounding_box is not None:
            pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
            rpy = np.clip(
                Rotation.from_quat(pose[3:]).as_euler("xyz"), self.rpy_bounding_box.low, self.rpy_bounding_box.high
            )
            pose[3:] = Rotation.from_euler("xyz", rpy).as_quat()
        return pose

    def get_im(self, quality: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Retrieve and process camera images from server.

        Downloads compressed images, decodes them, applies cropping/resizing,
        and converts from BGR to RGB format.

        Args:
            quality: JPEG compression quality (1-100) for server-side encoding

        Returns:
            Dictionary mapping camera names to processed RGB images (uint8)
        """
        try:
            url = f"{self.flask_server}/get_images"
            resp = self.session.post(
                url, timeout=self.request_timeout, params={"quality": quality} if quality is not None else None
            )
            if resp.status_code == 200:
                names = [n.strip() for n in resp.headers.get("Camera-Names", "").split(",") if n.strip()]
                body, images, offset = resp.content, {}, 0

                for name in names:
                    if offset + 4 > len(body):
                        break
                    (length,) = struct.unpack(">I", body[offset : offset + 4])
                    offset += 4
                    rgb = cv2.imdecode(np.frombuffer(body[offset : offset + length], np.uint8), cv2.IMREAD_COLOR)
                    offset += length

                    if rgb is not None:
                        cropped = self.image_crop[name](rgb) if name in self.image_crop else rgb
                        resized = cv2.resize(cropped, self.observation_space["images"][name].shape[:2][::-1])
                        images[name] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                return images
        except Exception as e:
            print(f"Error fetching cameras: {e}")
            input("Press enter to retry...")
            return self.get_im(quality)
        return {}

    def _send_eef_command(self, pos: np.ndarray, gripper_state: float) -> None:
        """Send end-effector pose and gripper command to server.

        Args:
            pos: End-effector pose [x,y,z,qx,qy,qz,qw]
            gripper_state: Target gripper joint position
        """
        try:
            data = {"eef_pose": np.array(pos).astype(np.float32).tolist(), "gripper_state": gripper_state}
            self._server_request("/move_eef", data)
        except Exception as e:
            print(f"Failed to send EEF command: {e}", file=sys.stderr)

    def _send_joint_command(self, pos: np.ndarray) -> None:
        """Send joint position command to server.

        Args:
            pos: Array of target joint positions
        """
        try:
            data = {"joint_pose": np.array(pos).astype(np.float32).tolist()}
            self._server_request("/move_joints", data)
        except Exception as e:
            print(f"Failed to send joint command: {e}", file=sys.stderr)

    def _reset_standard_pose(self) -> None:
        """Reset robot to standard initial pose via server request."""
        try:
            self._server_request("/reset_robot")
        except Exception as e:
            print(f"Failed to reset robot: {e}", file=sys.stderr)

    def _reset_isaacsim(self) -> None:
        """Reset Isaac Sim simulation environment via server request."""
        try:
            self._server_request("/reset_isaacsim")
        except Exception as e:
            print(f"Failed to reset isaacsim: {e}", file=sys.stderr)

    def _get_obs(self) -> Dict[str, Any]:
        """Construct complete observation dictionary from robot state and camera images.

        Returns:
            Observation dict with:
            - "state": Robot state (TCP pose/vel/force/torque, joint pos/vel)
            - "images": Camera images from all configured cameras
        """
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

    def compute_reward(self, ob: Dict[str, Any]) -> bool:
        """Base reward function (to be overridden by task-specific implementations).

        Args:
            ob: Environment observation dictionary

        Returns:
            Boolean indicating if task success condition is met
        """
        return False

    def go_to_reset(self) -> None:
        """Move robot to reset pose (random or standard).

        If random reset is enabled, adds random perturbations to the base reset pose
        within configured joint ranges before sending joint command.
        """
        self._update_currpos()
        if self.randomreset:
            reset_pose = self.reset_pose.copy()
            reset_pose[0] = np.random.uniform(-1 * self.random_joint_range[0], self.random_joint_range[0])
            reset_pose[1:-1] += np.random.uniform(
                -1 * self.random_joint_range[1], self.random_joint_range[1], (len(reset_pose) - 2,)
            )
            self._send_joint_command(reset_pose)
        else:
            self._reset_standard_pose()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], int, bool, bool, Dict[str, bool]]:
        """Execute one step of robot control.

        Processes action input, scales it, computes new end-effector pose,
        sends command to server, enforces timing, updates state, and computes
        reward/done status.

        Args:
            action: Normalized action array (7 DOF: 6 EEF pose, 1 gripper)

        Returns:
            Tuple containing:
            - observation: Current environment observation
            - reward: Integer reward (1 for success, 0 otherwise)
            - terminated: Whether episode has terminated (max length/reward/ESC)
            - truncated: Always False (no time truncation)
            - info: Dictionary with "succeed" key indicating task success
        """
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        next_eef_pos = self.curr_eef_poses_quat.copy()
        next_eef_pos[:3] = next_eef_pos[:3] + action[:3] * self.action_scale[0]
        next_eef_pos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1]) * Rotation.from_quat(self.curr_eef_poses_quat[3:])
        ).as_quat()

        gripper_action = float(self.curr_joint_positions[-1] + action[6] * self.action_scale[2])
        gripper_action = np.clip(gripper_action, self.gripper_limits[0], self.gripper_limits[1])
        next_eef_poses = self.clip_safety_box(next_eef_pos)

        self._send_eef_command(next_eef_poses, gripper_action)
        self.curr_path_length += 1

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()

        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate

        return ob, int(reward), done, False, {"succeed": reward}

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        """Reset environment to initial state.

        Moves robot to reset pose, resets simulation, and resets episode counters.

        Args:
            **kwargs: Additional reset arguments (unused)

        Returns:
            Tuple containing:
            - observation: Initial environment observation after reset
            - info: Dictionary with "succeed" key (always False for reset)
        """
        self.go_to_reset()
        time.sleep(3.5)
        self._reset_isaacsim()
        self.curr_path_length, self.terminate = 0, False
        self._update_currpos()
        return self._get_obs(), {"succeed": False}

    def close(self) -> None:
        """Cleanup environment resources.

        Closes HTTP session and stops keyboard listener to prevent resource leaks.
        """
        if hasattr(self, "session"):
            self.session.close()
        if hasattr(self, "listener"):
            self.listener.stop()

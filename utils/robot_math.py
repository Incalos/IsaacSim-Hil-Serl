import numpy as np
from scipy.spatial.transform import Rotation


def isaac_quat_to_scipy_quat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] -> [x, y, z, w]."""
    if quat.shape[0] != 4:
        raise ValueError(f"Quaternion must have 4 components, got {quat.shape[0]}")
    return np.array([quat[1], quat[2], quat[3], quat[0]])


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to Euler angles [yaw, pitch, roll] (degrees)."""
    r = Rotation.from_quat(quat)
    return r.as_euler("zyx", degrees=True)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles [yaw, pitch, roll] (degrees) to quaternion [x, y, z, w]."""
    r = Rotation.from_euler("zyx", euler, degrees=True)
    return r.as_quat()

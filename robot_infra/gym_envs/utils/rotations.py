from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion


def quat_2_euler(quat):
    # Standard conversion from Quaternion to Euler angles using the XYZ intrinsic sequence
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    # Unpack Euler angles; note the specific offset applied to the yaw component
    yaw, pitch, roll = xyz
    yaw = np.pi - yaw
    # Explicitly construct the Yaw (Z-axis) rotation matrix
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0, 0, 1.0],
        ]
    )
    # Explicitly construct the Pitch (Y-axis) rotation matrix
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    # Explicitly construct the Roll (X-axis) rotation matrix
    roll_matrix = np.array(
        [
            [1.0, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    # Combine the individual axis rotations into a single rotation matrix via dot product
    # This follows the sequence: R = R_yaw * R_pitch * R_roll
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    # Return the quaternion representation (w, x, y, z) from the combined matrix
    return Quaternion(matrix=rot_mat).elements

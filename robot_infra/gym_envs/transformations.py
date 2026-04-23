import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union, List


def construct_adjoint_matrix(tcp_pose: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Construct 6x6 adjoint matrix from TCP pose (position + quaternion rotation).

    Args:
        tcp_pose: TCP pose array with format [x, y, z, qx, qy, qz, qw] (position + quaternion)

    Returns:
        6x6 adjoint transformation matrix
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_transform_matrix(tcp_pose: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Construct 6x6 transform matrix from TCP pose (position + quaternion rotation).

    Args:
        tcp_pose: TCP pose array with format [x, y, z, qx, qy, qz, qw] (position + quaternion)

    Returns:
        6x6 transform matrix with rotation blocks in top-left and bottom-right
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    transform_matrix = np.zeros((6, 6))
    transform_matrix[:3, :3] = rotation
    transform_matrix[3:, 3:] = rotation
    return transform_matrix


def construct_homogeneous_matrix(tcp_pose: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Construct 4x4 homogeneous transformation matrix (position + quaternion rotation).

    Args:
        tcp_pose: TCP pose array with format [x, y, z, qx, qy, qz, qw] (position + quaternion)

    Returns:
        4x4 homogeneous transformation matrix
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T


def construct_adjoint_matrix_from_euler(tcp_pose: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Construct 6x6 adjoint matrix from TCP pose (position + Euler angle rotation).

    Args:
        tcp_pose: TCP pose array with format [x, y, z, rx, ry, rz] (position + xyz Euler angles)

    Returns:
        6x6 adjoint transformation matrix
    """
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix_from_euler(tcp_pose: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Construct 4x4 homogeneous transformation matrix (position + Euler angle rotation).

    Args:
        tcp_pose: TCP pose array with format [x, y, z, rx, ry, rz] (position + xyz Euler angles)

    Returns:
        4x4 homogeneous transformation matrix
    """
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T

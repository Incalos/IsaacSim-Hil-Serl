from scipy.spatial.transform import Rotation as R
import numpy as np


def construct_adjoint_matrix(tcp_pose):
    # Convert quaternion orientation (qx, qy, qz, qw) to 3x3 rotation matrix
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    # Create skew-symmetric matrix [p] for the translation vector to handle cross products
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    # Build 6x6 Adjoint matrix: $$Ad_T = \begin{bmatrix} R & 0 \\ [p]R & R \end{bmatrix}$$
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_transform_matrix(tcp_pose):
    # Extract rotation for delta-pose controllers that ignore translation in the 6x6 mapping
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    transform_matrix = np.zeros((6, 6))
    transform_matrix[:3, :3] = rotation
    transform_matrix[3:, 3:] = rotation
    return transform_matrix


def construct_homogeneous_matrix(tcp_pose):
    # Extract components to build a standard 4x4 SE(3) transformation matrix
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T


def construct_adjoint_matrix_from_euler(tcp_pose):
    # Generate rotation matrix using Euler angles (Intrinsic XYZ convention)
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    # Map spatial velocities using the Adjoint representation
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix_from_euler(tcp_pose):
    # Construct 4x4 homogeneous matrix specifically for Euler-based input poses
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T

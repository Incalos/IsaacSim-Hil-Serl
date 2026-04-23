import builtins
import numpy as np
import omni.graph.core as og
from typing import Any


def setup(db: og.Database) -> None:
    """Initialize per-instance state for robot state collection script node.

    Args:
        db: OmniGraph database instance for script node
    """
    state: Any = db.per_instance_state
    state.robot = getattr(builtins, "_GLOBAL_ROBOT_PRIM", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)


def compute(db: og.Database) -> None:
    """Main computation logic for collecting and processing robot state data.

    Args:
        db: OmniGraph database instance for script node
    """
    state: Any = db.per_instance_state

    forces_torques = state.robot.get_measured_joint_forces(joint_names=state.joint_names)
    forces_torques = forces_torques.squeeze(0).detach().cpu().numpy()

    joint_velocities = state.robot.get_joint_velocities(joint_names=state.joint_names)
    joint_velocities = joint_velocities.squeeze(0).detach().cpu().numpy()

    joint_efforts = state.robot.get_measured_joint_efforts()
    joint_efforts = joint_efforts.squeeze(0).detach().cpu().numpy()

    jacobians = state.robot.get_jacobians().detach().cpu().numpy()
    ee_link_idx = jacobians.shape[1] - 1
    jacobian_ee = jacobians[0, ee_link_idx, :, :]

    ee_vel = jacobian_ee @ joint_velocities

    jacobian_transpose_pinv = np.linalg.pinv(jacobian_ee.T)
    eef_wrench = jacobian_transpose_pinv @ joint_efforts

    db.outputs.measuredJointForces = forces_torques[:, :3].astype(np.float32, copy=False)
    db.outputs.measuredJointTorques = forces_torques[:, 3:].astype(np.float32, copy=False)
    db.outputs.measuredEEFVelocities = ee_vel.astype(np.float32, copy=False)
    db.outputs.measuredEEFWrenches = eef_wrench.astype(np.float32, copy=False)
    db.outputs.measuredEEFJacobians = jacobian_ee.astype(np.float32, copy=False)

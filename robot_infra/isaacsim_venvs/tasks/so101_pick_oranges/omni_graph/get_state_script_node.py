"""OmniGraph script node for robot state for the SO101 pick-oranges environment."""

import builtins

import numpy as np

import omni.graph.core as og


def setup(db: og.Database):
    state = db.per_instance_state

    # These globals are injected by the Isaac Sim graph/controller side.
    state.robot = getattr(builtins, "_GLOBAL_ROBOT_PRIM", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)


def compute(db: og.Database):
    state = db.per_instance_state

    # Joint forces/torques (per joint): [:3] forces, [3:] torques.
    forces_torques = state.robot.get_measured_joint_forces(joint_names=state.joint_names)
    forces_torques = forces_torques.squeeze(0).detach().cpu().numpy()

    # Joint velocities used to compute end-effector twist via Jacobian.
    joint_velocities = state.robot.get_joint_velocities(joint_names=state.joint_names)
    joint_velocities = joint_velocities.squeeze(0).detach().cpu().numpy()

    # Joint efforts (torques) used to estimate end-effector wrench.
    joint_efforts = state.robot.get_measured_joint_efforts()
    joint_efforts = joint_efforts.squeeze(0).detach().cpu().numpy()

    # Jacobians: (num_envs, num_links, 6, num_dofs). End-effector is typically the last link.
    jacobians = state.robot.get_jacobians().detach().cpu().numpy()
    ee_link_idx = jacobians.shape[1] - 1
    jacobian_ee = jacobians[0, ee_link_idx, :, :]

    # End-effector twist: [vx, vy, vz, wx, wy, wz].
    ee_vel = jacobian_ee @ joint_velocities

    # End-effector wrench estimate: F = (J^T)^+ * tau.
    jacobian_transpose_pinv = np.linalg.pinv(jacobian_ee.T)
    eef_wrench = jacobian_transpose_pinv @ joint_efforts

    db.outputs.measuredJointForces = forces_torques[:, :3].astype(np.float32, copy=False)
    db.outputs.measuredJointTorques = forces_torques[:, 3:].astype(np.float32, copy=False)

    db.outputs.measuredEEFVelocities = ee_vel.astype(np.float32, copy=False)
    db.outputs.measuredEEFWrenches = eef_wrench.astype(np.float32, copy=False)
    db.outputs.measuredEEFJacobians = jacobian_ee.astype(np.float32, copy=False)

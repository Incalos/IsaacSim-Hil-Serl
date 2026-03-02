import builtins
import numpy as np
import omni.graph.core as og


def setup(db: og.Database):
    # Initialize per-instance state
    state = db.per_instance_state
    # Get global robot prim and joint names injected by Isaac Sim
    state.robot = getattr(builtins, "_GLOBAL_ROBOT_PRIM", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)


def compute(db: og.Database):
    # Get per-instance state
    state = db.per_instance_state

    # Get joint forces and torques, convert to numpy array
    forces_torques = state.robot.get_measured_joint_forces(joint_names=state.joint_names)
    forces_torques = forces_torques.squeeze(0).detach().cpu().numpy()

    # Get joint velocities for end-effector twist calculation
    joint_velocities = state.robot.get_joint_velocities(joint_names=state.joint_names)
    joint_velocities = joint_velocities.squeeze(0).detach().cpu().numpy()

    # Get joint efforts for end-effector wrench estimation
    joint_efforts = state.robot.get_measured_joint_efforts()
    joint_efforts = joint_efforts.squeeze(0).detach().cpu().numpy()

    # Extract end-effector jacobian from robot jacobians
    jacobians = state.robot.get_jacobians().detach().cpu().numpy()
    ee_link_idx = jacobians.shape[1] - 1
    jacobian_ee = jacobians[0, ee_link_idx, :, :]

    # Calculate end-effector twist using jacobian and joint velocities
    ee_vel = jacobian_ee @ joint_velocities

    # Estimate end-effector wrench using pseudoinverse of jacobian transpose
    jacobian_transpose_pinv = np.linalg.pinv(jacobian_ee.T)
    eef_wrench = jacobian_transpose_pinv @ joint_efforts

    # Assign outputs with float32 type
    db.outputs.measuredJointForces = forces_torques[:, :3].astype(np.float32, copy=False)
    db.outputs.measuredJointTorques = forces_torques[:, 3:].astype(np.float32, copy=False)
    db.outputs.measuredEEFVelocities = ee_vel.astype(np.float32, copy=False)
    db.outputs.measuredEEFWrenches = eef_wrench.astype(np.float32, copy=False)
    db.outputs.measuredEEFJacobians = jacobian_ee.astype(np.float32, copy=False)

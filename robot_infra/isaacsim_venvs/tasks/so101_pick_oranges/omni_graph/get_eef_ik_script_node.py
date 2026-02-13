"""OmniGraph script node for end-effector inverse kinematics (IK) for the SO101 pick-oranges environment."""

import builtins

from curobo.types.math import Pose
import numpy as np
import torch

import omni.graph.core as og


def setup(db: og.Database):
    state = db.per_instance_state

    # These globals are injected by the Isaac Sim graph/controller side.
    state.ik_solver = getattr(builtins, "_GLOBAL_IK_SOLVER", None)
    state.tensor_args = getattr(builtins, "_GLOBAL_TENSOR_ARGS", None)
    state.graph_path = getattr(builtins, "_GLOBAL_GRAPH_PATH", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)

    if state.joint_names:
        state.num_joints = len(state.joint_names)
    else:
        state.num_joints = 0

    # Pre-allocate fixed-shape buffers to avoid per-tick allocations.
    if state.tensor_args is not None and hasattr(state.tensor_args, "device"):
        device = state.tensor_args.device
        dtype = state.tensor_args.dtype
        state.pos_buf = torch.zeros((1, 3), device=device, dtype=dtype)
        state.quat_buf = torch.zeros((1, 4), device=device, dtype=dtype)


def compute(db: og.Database):
    state = db.per_instance_state

    # Validate required dependencies are ready.
    if state.ik_solver is None or state.graph_path is None or state.tensor_args is None:
        return
    if not hasattr(state, "pos_buf") or not hasattr(state, "quat_buf"):
        return

    # Expected input layout: [x, y, z, qx, qy, qz, qw, gripper]
    eef_attr = og.Controller.attribute(state.graph_path + "/eef_cmd_sub.outputs:data")
    eef_data = eef_attr.get()
    if eef_data is None or len(eef_data) < 8:
        return

    with torch.no_grad():
        input_tensor = torch.as_tensor(eef_data, dtype=state.tensor_args.dtype)

        # Copy into preallocated buffers; reorder quaternion to cuRobo convention [w, x, y, z].
        state.pos_buf.copy_(input_tensor[:3].view(1, 3))
        quat_xyzw = input_tensor[3:7]
        state.quat_buf.copy_(quat_xyzw[[3, 0, 1, 2]].view(1, 4))

        goal_pose = Pose(position=state.pos_buf, quaternion=state.quat_buf)
        result = state.ik_solver.solve_single(goal_pose)

        full_commands = np.zeros(state.num_joints, dtype=np.float64)
        if result.success.item():
            solution_np = result.solution.detach().cpu().numpy().flatten()
            full_commands[:5] = solution_np[:5]
            full_commands[5] = eef_data[7]

            db.outputs.joint_names = state.joint_names
            db.outputs.position_command = full_commands
            db.outputs.velocity_command = np.full(state.num_joints, np.nan)
            db.outputs.effort_command = np.full(state.num_joints, np.nan)

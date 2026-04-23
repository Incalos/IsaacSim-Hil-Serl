import builtins
import numpy as np
import torch
import omni.graph.core as og
from typing import Any
from curobo.types.math import Pose


def setup(db: og.Database) -> None:
    """Initialize per-instance state for end-effector IK script node.

    Args:
        db: OmniGraph database instance for script node
    """
    state: Any = db.per_instance_state
    state.tensor_args = getattr(builtins, "_GLOBAL_TENSOR_ARGS", None)
    state.ik_solver = getattr(builtins, "_GLOBAL_IK_SOLVER", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)
    state.graph_path = getattr(builtins, "_GLOBAL_GRAPH_PATH", None)

    state.num_joints = len(state.joint_names) if state.joint_names else 0

    if state.tensor_args is not None and hasattr(state.tensor_args, "device"):
        device = state.tensor_args.device
        dtype = state.tensor_args.dtype
        state.pos_buf = torch.zeros((1, 3), device=device, dtype=dtype)
        state.quat_buf = torch.zeros((1, 4), device=device, dtype=dtype)


def compute(db: og.Database) -> None:
    """Main computation logic for end-effector IK solving and joint command generation.

    Args:
        db: OmniGraph database instance for script node
    """
    state: Any = db.per_instance_state

    if state.ik_solver is None or state.graph_path is None or state.tensor_args is None:
        return
    if not hasattr(state, "pos_buf") or not hasattr(state, "quat_buf"):
        return

    eef_attr = og.Controller.attribute(state.graph_path + "/eef_cmd_sub.outputs:data")
    eef_data = eef_attr.get()

    if eef_data is None or len(eef_data) < 8:
        return

    with torch.no_grad():
        input_tensor = torch.as_tensor(eef_data, dtype=state.tensor_args.dtype)
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

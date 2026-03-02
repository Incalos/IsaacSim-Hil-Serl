import builtins
from curobo.types.math import Pose
import numpy as np
import torch
import omni.graph.core as og


def setup(db: og.Database):
    # Initialize per-instance state
    state = db.per_instance_state

    # Get global variables injected by Isaac Sim
    state.ik_solver = getattr(builtins, "_GLOBAL_IK_SOLVER", None)
    state.tensor_args = getattr(builtins, "_GLOBAL_TENSOR_ARGS", None)
    state.graph_path = getattr(builtins, "_GLOBAL_GRAPH_PATH", None)
    state.joint_names = getattr(builtins, "_GLOBAL_JOINT_NAMES", None)

    # Calculate number of joints
    state.num_joints = len(state.joint_names) if state.joint_names else 0

    # Pre-allocate tensor buffers to avoid per-tick memory allocation
    if state.tensor_args is not None and hasattr(state.tensor_args, "device"):
        device = state.tensor_args.device
        dtype = state.tensor_args.dtype
        state.pos_buf = torch.zeros((1, 3), device=device, dtype=dtype)
        state.quat_buf = torch.zeros((1, 4), device=device, dtype=dtype)


def compute(db: og.Database):
    # Get per-instance state
    state = db.per_instance_state

    # Validate required dependencies are initialized
    if state.ik_solver is None or state.graph_path is None or state.tensor_args is None:
        return
    if not hasattr(state, "pos_buf") or not hasattr(state, "quat_buf"):
        return

    # Get end-effector command data from OmniGraph attribute
    eef_attr = og.Controller.attribute(state.graph_path + "/eef_cmd_sub.outputs:data")
    eef_data = eef_attr.get()

    # Validate input data length (expected: [x,y,z,qx,qy,qz,qw,gripper])
    if eef_data is None or len(eef_data) < 8:
        return

    # Disable gradient computation for inference
    with torch.no_grad():
        # Convert input data to tensor with matching dtype
        input_tensor = torch.as_tensor(eef_data, dtype=state.tensor_args.dtype)

        # Copy position data to preallocated buffer
        state.pos_buf.copy_(input_tensor[:3].view(1, 3))

        # Reorder quaternion from [x,y,z,w] to cuRobo's [w,x,y,z] convention and copy to buffer
        quat_xyzw = input_tensor[3:7]
        state.quat_buf.copy_(quat_xyzw[[3, 0, 1, 2]].view(1, 4))

        # Create target pose object for IK solver
        goal_pose = Pose(position=state.pos_buf, quaternion=state.quat_buf)

        # Solve IK for the target end-effector pose
        result = state.ik_solver.solve_single(goal_pose)

        # Initialize joint command array
        full_commands = np.zeros(state.num_joints, dtype=np.float64)

        # Update commands if IK solution is successful
        if result.success.item():
            # Convert tensor solution to numpy array
            solution_np = result.solution.detach().cpu().numpy().flatten()

            # Assign joint positions (first 5 joints from IK, 6th from gripper command)
            full_commands[:5] = solution_np[:5]
            full_commands[5] = eef_data[7]

            # Set output values for OmniGraph
            db.outputs.joint_names = state.joint_names
            db.outputs.position_command = full_commands
            db.outputs.velocity_command = np.full(state.num_joints, np.nan)
            db.outputs.effort_command = np.full(state.num_joints, np.nan)

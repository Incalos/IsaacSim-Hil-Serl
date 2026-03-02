import builtins
import torch
import omni.graph.core as og


# Initialize per-instance state with global environment and graph path
def setup(db: og.Database):
    state = db.per_instance_state
    state.base_env = getattr(builtins, "_GLOBAL_UNWRAP_ENV", None)
    state.graph_path = getattr(builtins, "_GLOBAL_GRAPH_PATH", None)


# Main computation logic for resetting IsaacSim environment
def compute(db: og.Database):
    state = db.per_instance_state

    # Get reset flag attribute from OmniGraph node
    flag_attr = og.Controller.attribute(state.graph_path + "/isaacsim_reset_sub.outputs:data")
    flag_data = flag_attr.get()

    # Trigger environment reset if flag is activated
    if flag_data:
        # Generate environment IDs for all available environments
        all_env_ids = torch.arange(state.base_env.num_envs, device=state.base_env.device, dtype=torch.long)

        # Apply reset event to all environments
        state.base_env.event_manager.apply(mode="reset", env_ids=all_env_ids, global_env_step_count=state.base_env.common_step_counter)

        # Reset flag to 0 after reset operation completes
        flag_attr.set(0)

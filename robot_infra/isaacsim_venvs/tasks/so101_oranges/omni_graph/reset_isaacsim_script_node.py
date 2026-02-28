"""OmniGraph script node for robot state for the SO101 pick-oranges environment."""

import builtins
import torch
import omni.graph.core as og


def setup(db: og.Database):
    state = db.per_instance_state
    state.base_env = getattr(builtins, "_GLOBAL_UNWRAP_ENV", None)
    state.graph_path = getattr(builtins, "_GLOBAL_GRAPH_PATH", None)


def compute(db: og.Database):
    state = db.per_instance_state

    flag_attr = og.Controller.attribute(state.graph_path + "/isaacsim_reset_sub.outputs:data")
    flag_data = flag_attr.get()
    
    if flag_data:
        all_env_ids = torch.arange(state.base_env.num_envs, device=state.base_env.device, dtype=torch.long)
        state.base_env.event_manager.apply(
            mode="reset", env_ids=all_env_ids, global_env_step_count=state.base_env.common_step_counter
        )
        flag_attr.set(0)

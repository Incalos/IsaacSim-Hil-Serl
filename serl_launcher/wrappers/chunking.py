from collections import deque
from typing import Optional, Dict, List, Any
import gymnasium
import numpy as np


def _tree_map_stack(d: Dict[str, Any]) -> Dict[str, Any]:
    """Stack list-valued leaves in a nested dictionary into numpy ndarrays.

    Args:
        d: Nested dictionary with list-valued leaves to be stacked.

    Returns:
        Dict with list-valued leaves converted to numpy arrays via stacking.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _tree_map_stack(v)
        elif isinstance(v, list):
            result[k] = np.stack(v)
        else:
            result[k] = v
    return result


def stack_obs(obs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert list of observation dicts to single dict with values stacked on axis 0.

    Args:
        obs: List of observation dicts with consistent keys and value shapes.

    Returns:
        Dict where each value is a numpy array stacked from input list.
    """
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return _tree_map_stack(dict_list)


def space_stack(space: gymnasium.Space, repeat: int) -> gymnasium.Space:
    """Create space repeating original along new leading dimension.

    Args:
        space: Original gymnasium Space to repeat.
        repeat: Number of repetitions for the new leading dimension.

    Returns:
        New gymnasium Space with repeated leading dimension.

    Raises:
        TypeError: If space type is not Box/Discrete/Dict.
    """
    if isinstance(space, gymnasium.spaces.Box):
        return gymnasium.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    if isinstance(space, gymnasium.spaces.Discrete):
        return gymnasium.spaces.MultiDiscrete([space.n] * repeat)
    if isinstance(space, gymnasium.spaces.Dict):
        return gymnasium.spaces.Dict({k: space_stack(v, repeat) for k, v in space.spaces.items()})
    raise TypeError(f"Unsupported space type: {type(space)}")


class ChunkingWrapper(gymnasium.Wrapper):
    """Wrapper for stacked observations and multi-step actions.

    Wraps env to return observations as length-obs_horizon stacks and execute
    act_exec_horizon steps per action (if specified).
    """

    def __init__(self, env: gymnasium.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon
        self.current_obs = deque(maxlen=self.obs_horizon)
        self.observation_space = space_stack(self.env.observation_space, self.obs_horizon)

        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(self.env.action_space, self.act_exec_horizon)

    def step(self, action: Any, *args) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute action sequence and return stacked observations.

        Args:
            action: Action/action sequence to execute.
            *args: Additional args for underlying env step.

        Returns:
            (stacked_obs, reward, done, trunc, info) from env.
        """
        act_exec_horizon = self.act_exec_horizon or 1
        action = [action] if self.act_exec_horizon is None else action
        assert len(action) >= act_exec_horizon, "Action length < execution horizon"

        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)

        return (stack_obs(self.current_obs), reward, done, trunc, info)

    def reset(self, **kwargs) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset env and initialize observation deque with full history.

        Args:
            **kwargs: Additional args for underlying env reset.

        Returns:
            (stacked_initial_obs, info) from env.
        """
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


def post_stack_obs(obs: Dict[str, np.ndarray], obs_horizon: int = 1) -> Dict[str, np.ndarray]:
    """Add batch dimension to flat observation dict values (obs_horizon=1 only).

    Args:
        obs: Observation dict with flat numpy array values.
        obs_horizon: Horizon steps (only 1 supported).

    Returns:
        Observation dict with added batch dimension.

    Raises:
        NotImplementedError: If obs_horizon != 1.
    """
    if obs_horizon != 1:
        raise NotImplementedError("Only obs_horizon=1 is supported for now")
    return {k: v[None] for k, v in obs.items()}

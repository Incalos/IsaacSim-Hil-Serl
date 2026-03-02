from collections import deque
from typing import Optional
import gymnasium as gym
import numpy as np


def _tree_map_stack(d):
    # Recursively stack lists in nested dict
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _tree_map_stack(v)
        elif isinstance(v, list):
            result[k] = np.stack(v)
        else:
            result[k] = v
    return result


def stack_obs(obs):
    # Convert list of observation dicts to dict of stacked arrays
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return _tree_map_stack(dict_list)


def space_stack(space: gym.Space, repeat: int):
    # Construct time-stacked Gym space from input space
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: space_stack(v, repeat) for k, v in space.spaces.items()})
    else:
        raise TypeError()


class ChunkingWrapper(gym.Wrapper):
    # Maintain observation history and execute action sequences
    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon
        self.current_obs = deque(maxlen=self.obs_horizon)
        # Stack observation space over time horizon
        self.observation_space = space_stack(self.env.observation_space, self.obs_horizon)

        # Configure action space (single-step or stacked)
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(self.env.action_space, self.act_exec_horizon)

    def step(self, action, *args):
        # Handle single action if no execution horizon is set
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        # Validate action sequence length
        assert len(action) >= act_exec_horizon

        # Execute action sequence and accumulate observations
        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)

        return stack_obs(self.current_obs), reward, done, trunc, info

    def reset(self, **kwargs):
        # Reset environment and initialize observation history
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


def post_stack_obs(obs, obs_horizon=1):
    # Wrap single-step observation with leading time dimension
    if obs_horizon != 1:
        raise NotImplementedError("Only obs_horizon=1 is supported for now")
    obs = {k: v[None] for k, v in obs.items()}
    return obs

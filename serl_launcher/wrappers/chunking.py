from collections import deque
from typing import Optional
import gymnasium as gym
import numpy as np


def _tree_map_stack(d):
    # Recursively stack lists contained in a nested dict
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
    # Convert a list of observation dicts into a dict of stacked arrays
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return _tree_map_stack(dict_list)


def space_stack(space: gym.Space, repeat: int):
    # Construct a Gym space that represents a time-stacked version of the input space
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
    # Wrapper that maintains an observation history and optionally executes action sequences
    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon
        # Deque storing the last obs_horizon observations
        self.current_obs = deque(maxlen=self.obs_horizon)
        # Observation space is stacked over the time horizon
        self.observation_space = space_stack(self.env.observation_space, self.obs_horizon)
        # Action space can be either single-step or stacked over an execution horizon
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(self.env.action_space, self.act_exec_horizon)

    def step(self, action, *args):
        # If no execution horizon is set, treat the input as a single action
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1
        # Ensure we received at least as many actions as the execution horizon
        assert len(action) >= act_exec_horizon
        # Execute a sequence of actions and accumulate observations
        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
        return stack_obs(self.current_obs), reward, done, trunc, info

    def reset(self, **kwargs):
        # Reset the environment and fill the history with the first observation
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


def post_stack_obs(obs, obs_horizon=1):
    # Wrap a single-step observation dict with a leading time dimension
    if obs_horizon != 1:
        raise NotImplementedError("Only obs_horizon=1 is supported for now")
    obs = {k: v[None] for k, v in obs.items()}
    return obs

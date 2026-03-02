import wandb
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
import time
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import cloudpickle as pickle
from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym


class WandBLogger(object):
    # Get default configuration for WandB logger with optional updates
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = "JaxCQL"
        config.project = ""
        config.output_dir = "/tmp/JaxCQL"
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        config.entity = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    # Initialize WandB logger with config and experiment variant
    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        # Generate unique experiment ID if not provided
        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        # Add prefix to project name if specified
        if self.config.prefix != "":
            self.config.project = "{}--{}".format(self.config.prefix, self.config.project)

        # Set up output directory (temp dir if empty, else create experiment subdir)
        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        # Copy variant and add hostname if missing
        self._variant = copy(variant)
        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        # Add random delay if configured
        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        # Initialize WandB run
        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            entity=config.entity,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode="online" if self.config.online else "offline",
        )

    # Log metrics to WandB
    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    # Save object to pickle file in output directory
    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
            pickle.dump(obj, fout)

    # Property for experiment ID
    @property
    def experiment_id(self):
        return self.config.experiment_id

    # Property for experiment variant
    @property
    def variant(self):
        return self.config.variant

    # Property for output directory
    @property
    def output_dir(self):
        return self.config.output_dir


class RecordEpisodeStatistics(gym.Wrapper, gym.utils.RecordConstructorArgs):
    # Initialize episode statistics tracker wrapper
    def __init__(self, env: gym.Env, deque_size: int = 100):
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        super().__init__(env)

        # Detect vectorized environment properties
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        # Initialize episode tracking variables
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    # Reset environment and episode statistics
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_start_times = np.full(self.num_envs, time.perf_counter(), dtype=np.float32)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    # Step through environment and update episode statistics
    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.env.step(action)

        # Validate info type (prevent wrapper order issues)
        assert isinstance(infos, dict), f"`info` dtype is {type(infos)} while supported dtype is `dict`."

        # Update cumulative returns and lengths
        self.episode_returns += rewards
        self.episode_lengths += 1

        # Check for completed episodes (terminated or truncated)
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)

        # Process completed episodes
        if num_dones:
            # Prevent duplicate episode stats in info
            if "episode" in infos or "_episode" in infos:
                raise ValueError("Attempted to add episode stats when they already exist")

            # Add episode stats to info dict
            infos["episode"] = {
                "r": np.where(dones, self.episode_returns, 0.0),
                "l": np.where(dones, self.episode_lengths, 0),
                "t": np.where(
                    dones,
                    np.round(time.perf_counter() - self.episode_start_times, 6),
                    0.0,
                ),
            }

            # Add vector env episode flag if needed
            if self.is_vector_env:
                infos["_episode"] = np.where(dones, True, False)

            # Update episode buffers and counters
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones

            # Reset tracking for completed environments
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()

        return observations, rewards, terminations, truncations, infos

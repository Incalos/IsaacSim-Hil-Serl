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
from typing import Optional, Dict, Any, Deque, Tuple
import numpy as np
import gymnasium
from gymnasium import Env
from gymnasium.utils import RecordConstructorArgs


class WandBLogger:

    @staticmethod
    def get_default_config(updates: Optional[Dict[str, Any]] = None) -> ConfigDict:
        """
        Get default WandB logger configuration

        Args:
            updates: Optional dictionary to override default config values

        Returns:
            ConfigDict with logger configuration
        """
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

    def __init__(self, config: Dict[str, Any], variant: Dict[str, Any]) -> None:
        """
        Initialize WandB logger

        Args:
            config: Logger configuration dictionary
            variant: Experiment variant/hyperparameters to log
        """
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            self.config.project = "{}--{}".format(self.config.prefix, self.config.project)

        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)
        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            entity=config.entity,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            mode="online" if self.config.online else "offline",
        )

    def log(self, *args, **kwargs) -> None:
        """Proxy method to log data to WandB"""
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj: Any, filename: str) -> None:
        """
        Save object to pickle file in logger output directory

        Args:
            obj: Object to serialize
            filename: Name of pickle file
        """
        with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self) -> str:
        """Unique experiment ID (read-only)"""
        return self.config.experiment_id

    @property
    def variant(self) -> Dict[str, Any]:
        """Experiment variant configuration (read-only)"""
        return self.config.variant

    @property
    def output_dir(self) -> str:
        """Output directory path (read-only)"""
        return self.config.output_dir


class RecordEpisodeStatistics(gymnasium.Wrapper, RecordConstructorArgs):

    def __init__(self, env: Env, deque_size: int = 100) -> None:
        """
        Wrapper to record episode statistics (returns, lengths, times)

        Args:
            env: Gymnasium environment to wrap
            deque_size: Maximum size of statistic queues (for rolling window)
        """
        RecordConstructorArgs.__init__(self, deque_size=deque_size)
        super().__init__(env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: Optional[np.ndarray] = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue: Deque[float] = deque(maxlen=deque_size)
        self.length_queue: Deque[int] = deque(maxlen=deque_size)

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment and reinitialize episode tracking

        Args:
            **kwargs: Additional reset arguments for the environment

        Returns:
            Initial observations and info dictionary
        """
        obs, info = self.env.reset(**kwargs)
        self.episode_start_times = np.full(self.num_envs, time.perf_counter(), dtype=np.float32)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action: Any) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute environment step and update episode statistics

        Args:
            action: Action to take in the environment

        Returns:
            (observations, rewards, terminations, truncations, infos) with episode stats added

        Raises:
            ValueError: If episode stats already exist in info dict
            AssertionError: If info is not a dictionary
        """
        observations, rewards, terminations, truncations, infos = self.env.step(action)

        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."

        self.episode_returns += rewards
        self.episode_lengths += 1

        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)

        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError("Attempted to add episode stats when they already exist")
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)

            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones

            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()

        return observations, rewards, terminations, truncations, infos

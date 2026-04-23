import gymnasium
import torch
import pickle as pkl
import numpy as np
from threading import Lock
from typing import Iterable
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer
from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    """Thread-safe replay buffer implementing DataStoreBase interface.

    Args:
        observation_space: Gymnasium observation space specification
        action_space: Gymnasium action space specification
        capacity: Maximum number of transitions to store
        device: Device to store buffer tensors (default: "cpu")
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        capacity: int,
        device: str = "cpu",
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity, device=device)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        """Thread-safe wrapper for base class insert method.

        Args:
            *args: Positional arguments for base insert method
            **kwargs: Keyword arguments for base insert method
        """
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Thread-safe wrapper for base class sample method.

        Args:
            *args: Positional arguments for base sample method
            **kwargs: Keyword arguments for base sample method

        Returns:
            Sampled batch data from replay buffer
        """
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self) -> int:
        """Return index of the most recently inserted data entry.

        Returns:
            Insert index of the latest data entry
        """
        return self._insert_index

    def get_latest_data(self, from_id: int):
        """Retrieve data entries from specified starting ID (placeholder).

        Args:
            from_id: Starting index for data retrieval

        Raises:
            NotImplementedError: Always raised (placeholder implementation)
        """
        raise NotImplementedError("TODO")


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    """Thread-safe memory-efficient replay buffer implementing DataStoreBase.

    Args:
        observation_space: Gymnasium observation space specification
        action_space: Gymnasium action space specification
        capacity: Maximum number of transitions to store
        image_keys: Keys for image/pixel observation data (default: ("image",))
        device: Device to store buffer tensors (default: "cpu")
        **kwargs: Additional arguments for MemoryEfficientReplayBuffer
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        device: str = "cpu",
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            pixel_keys=image_keys,
            device=device,
            **kwargs,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        """Thread-safe wrapper for base class insert method.

        Args:
            *args: Positional arguments for base insert method
            **kwargs: Keyword arguments for base insert method
        """
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Thread-safe wrapper for base class sample method.

        Args:
            *args: Positional arguments for base sample method
            **kwargs: Keyword arguments for base sample method

        Returns:
            Sampled batch data from memory-efficient replay buffer
        """
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self) -> int:
        """Return index of the most recently inserted data entry.

        Returns:
            Insert index of the latest data entry
        """
        return self._insert_index

    def get_latest_data(self, from_id: int):
        """Retrieve data entries from specified starting ID (placeholder).

        Args:
            from_id: Starting index for data retrieval

        Raises:
            NotImplementedError: Always raised (placeholder implementation)
        """
        raise NotImplementedError("TODO")


def populate_data_store(data_store: DataStoreBase, demos_path: str, device: str = "cpu") -> DataStoreBase:
    """Populate data store with demonstration transitions from pickle files.

    Args:
        data_store: Target data store instance to populate
        demos_path: List of paths to demonstration pickle files
        device: Device to transfer transition tensors to (default: "cpu")

    Returns:
        Populated data store instance
    """
    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                if isinstance(transition, dict):
                    transition = {
                        k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v)
                        for k, v in transition.items()
                    }
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
    device: str = "cpu",
) -> DataStoreBase:
    """Populate data store with z-axis only state features from demonstrations.

    Args:
        data_store: Target data store instance to populate
        demos_path: List of paths to demonstration pickle files
        device: Device to transfer transition tensors to (default: "cpu")

    Returns:
        Populated data store instance with z-axis only state features
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                state = torch.from_numpy(tmp["observations"]["state"]).to(device)
                next_state = torch.from_numpy(tmp["next_observations"]["state"]).to(device)
                tmp["observations"]["state"] = torch.cat([state[:, :4], state[:, 6:7], state[:, 10:]], dim=-1)
                tmp["next_observations"]["state"] = torch.cat(
                    [next_state[:, :4], next_state[:, 6:7], next_state[:, 10:]], dim=-1
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store

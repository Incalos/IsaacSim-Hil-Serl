from threading import Lock
from typing import Iterable
import gymnasium as gym
import torch
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer
from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int, device: str = "cpu"):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity, device=device)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # Thread-safe insertion method for replay buffer
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # Thread-safe sampling method for replay buffer
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # Get index of the latest inserted data entry
    def latest_data_id(self) -> int:
        return self._insert_index

    # Retrieve data entries from specified ID onwards (not implemented)
    def get_latest_data(self, from_id: int):
        raise NotImplementedError("TODO")


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):

    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            capacity: int,
            image_keys: Iterable[str] = ("image",),
            device: str = "cpu",
            **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(self, observation_space, action_space, capacity, pixel_keys=image_keys, device=device, **kwargs)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # Thread-safe insertion method for memory-efficient replay buffer
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # Thread-safe sampling method for memory-efficient replay buffer
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(*args, **kwargs)

    # Get index of the latest inserted data entry
    def latest_data_id(self) -> int:
        return self._insert_index

    # Retrieve data entries from specified ID onwards (not implemented)
    def get_latest_data(self, from_id: int):
        raise NotImplementedError("TODO")


def populate_data_store(data_store: DataStoreBase, demos_path: str, device: str = "cpu") -> DataStoreBase:
    import pickle as pkl
    import numpy as np

    # Iterate through all demonstration file paths
    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            # Process each transition in demonstration data
            for transition in demo:
                if isinstance(transition, dict):
                    # Convert numpy arrays to torch tensors and move to target device
                    transition = {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in transition.items()}
                data_store.insert(transition)
        # Print current number of loaded transitions
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(data_store: DataStoreBase, demos_path: str, device: str = "cpu") -> DataStoreBase:
    import pickle as pkl
    from copy import deepcopy

    # Iterate through all demonstration file paths
    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            # Process each transition in demonstration data
            for transition in demo:
                tmp = deepcopy(transition)
                # Convert state arrays to torch tensors and move to target device
                state = torch.from_numpy(tmp["observations"]["state"]).to(device)
                next_state = torch.from_numpy(tmp["next_observations"]["state"]).to(device)

                # Keep only z-axis related state features (remove x/y cartesian coordinates)
                tmp["observations"]["state"] = torch.cat(
                    [
                        state[:, :4],
                        state[:, 6:7],
                        state[:, 10:],
                    ],
                    dim=-1,
                )

                tmp["next_observations"]["state"] = torch.cat(
                    [
                        next_state[:, :4],
                        next_state[:, 6:7],
                        next_state[:, 10:],
                    ],
                    dim=-1,
                )

                data_store.insert(tmp)
        # Print current number of loaded transitions
        print(f"Loaded {len(data_store)} transitions.")
    return data_store

import collections
import torch
import gymnasium
import numpy as np
from typing import Optional, Union
from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gymnasium.Space,
    capacity: int,
    device: torch.device,
) -> Union[torch.Tensor, DatasetDict]:
    """Initialize replay buffer storage structure based on observation space type.

    Args:
        obs_space: Gymnasium observation space specification
        capacity: Maximum number of transitions to store
        device: Device to store buffer tensors

    Returns:
        Initialized tensor/dictionary structure for replay storage

    Raises:
        TypeError: If unsupported observation space type is provided
    """
    if isinstance(obs_space, gymnasium.spaces.Box):
        return torch.from_numpy(np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)).to(device)
    elif isinstance(obs_space, gymnasium.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity, device)
        return data_dict
    else:
        raise TypeError(f"Unsupported space type: {type(obs_space)}")


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int):
    """Recursively insert transition data into replay buffer storage.

    Args:
        dataset_dict: Target replay buffer storage structure
        data_dict: Transition data to insert
        insert_index: Buffer index to insert data at

    Raises:
        TypeError: If unsupported data type is found in transition data
    """
    if isinstance(dataset_dict, (torch.Tensor, np.ndarray)):
        if isinstance(data_dict, np.ndarray):
            dataset_dict[insert_index] = torch.from_numpy(data_dict).to(dataset_dict.device)
        elif isinstance(data_dict, np.generic):
            dataset_dict[insert_index] = data_dict.item()
        else:
            dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError(f"Unsupported type: {type(dataset_dict)}")


class ReplayBuffer(Dataset):
    """Replay buffer for storing and sampling RL transition data.

    Args:
        observation_space: Gymnasium observation space specification
        action_space: Gymnasium action space specification
        capacity: Maximum number of transitions to store
        next_observation_space: Next observation space (defaults to observation_space)
        include_next_actions: Whether to store next actions (default: False)
        include_label: Whether to store label data (default: False)
        include_grasp_penalty: Whether to store grasp penalty values (default: False)
        device: Device to store buffer tensors (default: "cpu")
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        capacity: int,
        next_observation_space: Optional[gymnasium.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        if next_observation_space is None:
            next_observation_space = observation_space
        observation_data = _init_replay_dict(observation_space, capacity, self.device)
        next_observation_data = _init_replay_dict(next_observation_space, capacity, self.device)
        dataset_dict = {
            "observations": observation_data,
            "next_observations": next_observation_data,
            "actions": torch.empty((capacity, *action_space.shape), dtype=torch.float32, device=self.device),
            "rewards": torch.empty((capacity,), dtype=torch.float32, device=self.device),
            "masks": torch.empty((capacity,), dtype=torch.float32, device=self.device),
            "dones": torch.empty((capacity,), dtype=torch.bool, device=self.device),
        }
        if include_next_actions:
            dataset_dict["next_actions"] = torch.empty(
                (capacity, *action_space.shape), dtype=torch.float32, device=self.device
            )
            dataset_dict["next_intvn"] = torch.empty((capacity,), dtype=torch.bool, device=self.device)
        if include_label:
            dataset_dict["labels"] = torch.empty((capacity,), dtype=torch.long, device=self.device)
        if include_grasp_penalty:
            dataset_dict["grasp_penalty"] = torch.empty((capacity,), dtype=torch.float32, device=self.device)
        super().__init__(dataset_dict)
        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        """Return current number of elements in replay buffer.

        Returns:
            Current size of the replay buffer
        """
        return self._size

    def insert(self, data_dict: DatasetDict):
        """Insert a single transition data into replay buffer.

        Args:
            data_dict: Transition data dictionary to insert
        """
        if isinstance(data_dict, dict):
            data_dict = {
                k: (torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v)
                for k, v in data_dict.items()
            }
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        """Get iterator for sampling batches with pre-fetching and device transfer.

        Args:
            queue_size: Number of batches to pre-fetch (default: 2)
            sample_args: Arguments to pass to sample method (default: {})
            device: Device to transfer batches to (defaults to buffer device)

        Yields:
            Sampled batch data transferred to target device
        """
        if device is None:
            device = self.device
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                if isinstance(data, dict):
                    data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        """Download a range of data from buffer by index range.

        Args:
            from_idx: Start index of data range
            to_idx: End index of data range (exclusive)

        Returns:
            Tuple containing end index and downloaded data dictionary
        """
        indices = torch.arange(from_idx, to_idx, device=self.device)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        """Iterator to download all data from buffer sequentially.

        Yields:
            Batches of data covering entire buffer

        Raises:
            RuntimeError: If download index exceeds buffer size
        """
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch

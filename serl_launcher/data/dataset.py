from typing import Dict, Iterable, Optional, Tuple, Union
import torch
import numpy as np
from gymnasium.utils import seeding

# Define nested data type for dataset elements
DataType = Union[torch.Tensor, np.ndarray, Dict[str, "DataType"]]
# Define dataset structure as dictionary of DataTypes
DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    # Recursively verify all elements have consistent length
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError(f"Unsupported type: {type(v)}")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: Union[np.ndarray, torch.Tensor]) -> DatasetDict:
    # Recursively select subset of dataset using index array
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        elif isinstance(v, torch.Tensor):
            new_v = v[torch.as_tensor(index, device=v.device)]
        else:
            raise TypeError(f"Unsupported type: {type(v)}")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(dataset_dict: Union[np.ndarray, torch.Tensor, DatasetDict], indx: Union[np.ndarray, torch.Tensor]) -> DatasetDict:
    # Recursively sample elements from dataset using indices
    if isinstance(dataset_dict, (np.ndarray, torch.Tensor)):
        if isinstance(dataset_dict, np.ndarray):
            return dataset_dict[indx]
        else:
            return dataset_dict[torch.as_tensor(indx, device=dataset_dict.device)]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError(f"Unsupported type: {type(dataset_dict)}")
    return batch


class Dataset(object):
    # Dataset class for handling reinforcement learning data with sampling/splitting/filtering
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None, device: str = "cpu"):
        self.device = torch.device(device)
        # Convert numpy arrays to torch tensors and move to target device
        self.dataset_dict = {k: (torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v) for k, v in dataset_dict.items()}
        # Validate and store dataset length
        self.dataset_len = _check_lengths(self.dataset_dict)

        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        # Lazy initialization of random number generator
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        # Set random seed for reproducible sampling
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        # Return total number of elements in dataset
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict:
        # Sample batch of data with optional key filtering and custom indices
        if indx is None:
            # Generate random indices if not provided
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)
            indx = torch.from_numpy(indx).to(self.device)

        batch = {}
        if keys is None:
            keys = self.dataset_dict.keys()

        # Extract requested keys using generated indices
        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return batch

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        # Split dataset into train/test subsets by random permutation
        assert 0 < ratio < 1

        index = torch.randperm(len(self), device=self.device)
        split_idx = int(self.dataset_len * ratio)

        train_index = index[:split_idx]
        test_index = index[split_idx:]

        # Create new Dataset instances for train/test subsets
        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict, device=self.device), Dataset(test_dataset_dict, device=self.device)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        # Calculate episode boundaries and cumulative returns per episode
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        rewards = self.dataset_dict["rewards"]
        dones = self.dataset_dict["dones"]

        # Convert to numpy for easy iteration
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()

        # Iterate through timesteps to find episode boundaries and compute returns
        for i in range(len(self)):
            episode_return += rewards[i]

            if dones[i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(self, take_top: Optional[float] = None, threshold: Optional[float] = None):
        # Filter dataset to keep only trajectories above return threshold
        assert (take_top is None and threshold is not None) or (take_top is not None and threshold is None)

        episode_starts, episode_ends, episode_returns = self._trajectory_boundaries_and_returns()

        # Calculate threshold from percentile if take_top is specified
        if take_top is not None:
            threshold = np.percentile(episode_returns, 100 - take_top)

        # Create boolean mask for timesteps in qualified trajectories
        bool_indx = torch.full((len(self),), False, dtype=torch.bool, device=self.device)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i]:episode_ends[i]] = True

        # Apply mask and update dataset
        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)
        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        # Normalize reward values to [0, scaling] range based on episode returns
        _, _, episode_returns = self._trajectory_boundaries_and_returns()
        # Normalize rewards tensor/numpy array
        if isinstance(self.dataset_dict["rewards"], torch.Tensor):
            self.dataset_dict["rewards"] /= max(episode_returns) - min(episode_returns)
            self.dataset_dict["rewards"] *= scaling
        else:
            self.dataset_dict["rewards"] /= np.max(episode_returns) - np.min(episode_returns)
            self.dataset_dict["rewards"] *= scaling

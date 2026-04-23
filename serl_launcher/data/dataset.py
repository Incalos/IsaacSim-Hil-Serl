import torch
import numpy as np
from typing import Dict, Iterable, Optional, Tuple, Union
from gymnasium.utils import seeding

DataType = Union[torch.Tensor, np.ndarray, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    """Validate all elements in dataset have consistent sequence lengths.

    Args:
        dataset_dict: Nested dictionary containing dataset tensors/arrays
        dataset_len: Pre-computed dataset length for validation (optional)

    Returns:
        Validated consistent dataset length

    Raises:
        TypeError: If unsupported data type is found in dataset
        AssertionError: If inconsistent lengths are detected
    """
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
    """Extract subset of dataset using provided index array/tensor.

    Args:
        dataset_dict: Original dataset dictionary to subset
        index: Indices to select from the dataset

    Returns:
        New dataset dictionary containing only selected indices

    Raises:
        TypeError: If unsupported data type is found in dataset
    """
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


def _sample(
    dataset_dict: Union[np.ndarray, torch.Tensor, DatasetDict],
    indx: Union[np.ndarray, torch.Tensor],
) -> DatasetDict:
    """Recursively sample elements from dataset structure using indices.

    Args:
        dataset_dict: Dataset structure to sample from (tensor/array/dict)
        indx: Indices to sample from the dataset

    Returns:
        Sampled subset of the dataset

    Raises:
        TypeError: If unsupported data type is found in dataset
    """
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
    """Core dataset class for RL data management (sampling/splitting/filtering/normalization).

    Args:
        dataset_dict: Nested dictionary containing RL transition data
        seed: Random seed for reproducible sampling (optional)
        device: Device to store dataset tensors (default: "cpu")
    """

    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.dataset_dict = {
            k: (torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v)
            for k, v in dataset_dict.items()
        }
        self.dataset_len = _check_lengths(self.dataset_dict)
        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        """Lazy initialization of random number generator.

        Returns:
            Initialized numpy random state generator
        """
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        """Set seed for reproducible random sampling.

        Args:
            seed: Random seed value (optional)

        Returns:
            List containing the initialized seed value
        """
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        """Return total number of timesteps in dataset.

        Returns:
            Total length of the dataset
        """
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict:
        """Sample batch of data from dataset using random or specified indices.

        Args:
            batch_size: Number of samples to retrieve
            keys: Subset of dataset keys to include in batch (optional)
            indx: Pre-specified indices to sample (optional)

        Returns:
            Dictionary containing sampled batch data
        """
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)
            indx = torch.from_numpy(indx).to(self.device)
        if keys is None:
            keys = self.dataset_dict.keys()
        batch = {}
        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]
        return batch

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        """Split dataset into train/test subsets by specified ratio.

        Args:
            ratio: Split ratio (0 < ratio < 1) for train dataset size

        Returns:
            Tuple of train and test Dataset instances

        Raises:
            AssertionError: If ratio is not between 0 and 1
        """
        assert 0 < ratio < 1
        index = torch.randperm(len(self), device=self.device)
        split_idx = int(self.dataset_len * ratio)
        train_index = index[:split_idx]
        test_index = index[split_idx:]
        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict, device=self.device), Dataset(test_dataset_dict, device=self.device)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        """Calculate episode boundaries and cumulative returns per episode.

        Returns:
            Tuple containing:
                - List of episode start indices
                - List of episode end indices
                - List of cumulative returns per episode
        """
        episode_starts = [0]
        episode_ends = []
        episode_return = 0
        episode_returns = []
        rewards = self.dataset_dict["rewards"]
        dones = self.dataset_dict["dones"]
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
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
        """Filter high-performing trajectories by return threshold or top percentile.

        Args:
            take_top: Percentage of top-performing trajectories to keep (0-100)
            threshold: Minimum return threshold for trajectory inclusion

        Raises:
            AssertionError: If both/neither take_top and threshold are specified
        """
        assert (take_top is None and threshold is not None) or (take_top is not None and threshold is None)
        episode_starts, episode_ends, episode_returns = self._trajectory_boundaries_and_returns()
        if take_top is not None:
            threshold = np.percentile(episode_returns, 100 - take_top)
        bool_indx = torch.full((len(self),), False, dtype=torch.bool, device=self.device)
        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i] : episode_ends[i]] = True
        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)
        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        """Normalize reward values to [0, scaling] range using min/max episode returns.

        Args:
            scaling: Maximum value for normalized rewards (default: 1000)
        """
        _, _, episode_returns = self._trajectory_boundaries_and_returns()
        if isinstance(self.dataset_dict["rewards"], torch.Tensor):
            self.dataset_dict["rewards"] /= max(episode_returns) - min(episode_returns)
            self.dataset_dict["rewards"] *= scaling
        else:
            self.dataset_dict["rewards"] /= np.max(episode_returns) - np.min(episode_returns)
            self.dataset_dict["rewards"] *= scaling

from collections import defaultdict
import imageio
import torch
import numpy as np
import wandb


def concat_batches(offline_batch, online_batch, axis=1):
    """Concatenate two batches along specified axis"""
    batch = defaultdict(list)

    if isinstance(offline_batch, dict) and isinstance(online_batch, dict):
        for k, v in offline_batch.items():
            if isinstance(v, dict):
                batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
            else:
                if isinstance(v, torch.Tensor) and isinstance(online_batch[k], torch.Tensor):
                    batch[k] = torch.cat((v, online_batch[k]), dim=axis)
                elif isinstance(v, np.ndarray) and isinstance(online_batch[k], np.ndarray):
                    batch[k] = np.concatenate((v, online_batch[k]), axis=axis)
                else:
                    raise TypeError(f"Unsupported type for concatenation: {type(v)} and {type(online_batch[k])}")
    return batch


def load_recorded_video(video_path: str):
    """Load and convert video for wandb logging"""
    video = np.array(imageio.mimread(video_path, "MP4")).transpose((0, 3, 1, 2))
    assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"
    return wandb.Video(video, fps=20)


def _unpack(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation
    """
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            if isinstance(batch["observations"][pixel_key], torch.Tensor):
                obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
                next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

                obs = dict(batch["observations"])
                obs[pixel_key] = obs_pixels

                next_obs = dict(batch["next_observations"])
                next_obs[pixel_key] = next_obs_pixels

                batch = dict(batch)
                batch["observations"] = obs
                batch["next_observations"] = next_obs

    return batch

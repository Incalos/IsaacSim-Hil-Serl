import os
import pickle as pkl
import requests
from collections import defaultdict
from tqdm import tqdm
import imageio
import torch
import numpy as np
import wandb
from typing import Dict, Any, Tuple


def concat_batches(offline_batch: Dict[str, Any], online_batch: Dict[str, Any], axis: int = 1) -> Dict[str, Any]:
    """
    Concatenate two batches of training data along specified axis

    Args:
        offline_batch: First batch (offline data)
        online_batch: Second batch (online data)
        axis: Axis to concatenate along

    Returns:
        Concatenated batch with same structure as input batches

    Raises:
        TypeError: If unsupported data types are encountered
    """
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


def load_recorded_video(video_path: str) -> wandb.Video:
    """
    Load video file and convert to WandB-compatible format

    Args:
        video_path: Path to MP4 video file

    Returns:
        WandB Video object ready for logging

    Raises:
        AssertionError: If video shape is not (T, C, H, W) with C=3
    """
    video = np.array(imageio.mimread(video_path, "MP4")).transpose((0, 3, 1, 2))
    assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"
    return wandb.Video(video, fps=20)


def _unpack(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split combined observation/next_observation pixel data to optimize CPU-GPU transfer

    Args:
        batch: Training batch with combined pixel observations

    Returns:
        Batch with separate observation/next_observation pixel data
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


def load_resnet10_params(agent: Any, image_keys: Tuple[str, ...] = ("image",), public: bool = True) -> Any:
    """
    Load pretrained ResNet-10 parameters and assign to agent's visual encoder

    Args:
        agent: Agent instance with visual encoder
        image_keys: Keys for image encoders to update
        public: If True, download weights from public repository; else load from local

    Returns:
        Agent with updated pretrained ResNet-10 encoder weights

    Raises:
        RuntimeError: If weight download fails
    """
    file_name = "resnet10_params.pkl"

    if not public:
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)

        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")

            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception("Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)

    encoder_params = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in encoder_params.items()}

    param_count = sum(p.numel() for p in encoder_params.values() if isinstance(p, torch.Tensor))
    print(f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K")

    for image_key in image_keys:
        encoder_name = f"encoder_{image_key}"
        if encoder_name in agent.actor.encoder.state_dict():
            encoder_state = agent.actor.encoder.state_dict()
            for k, v in encoder_params.items():
                if k in encoder_state:
                    encoder_state[k].copy_(v)
                    print(f"replaced {k} in pretrained_encoder")

            agent.actor.encoder.load_state_dict(encoder_state)
            if hasattr(agent, "critic") and hasattr(agent.critic, "encoder"):
                agent.critic.encoder.load_state_dict(encoder_state)

    return agent

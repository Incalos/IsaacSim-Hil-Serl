from collections import OrderedDict
from typing import Dict, List, Optional, Any
import gymnasium
import numpy as np


class VideoWrapper(gymnasium.Wrapper):
    """Wrapper to capture observation frames and generate video arrays.

    Collects frames from non-'state' observation keys and provides methods to
    retrieve/combine frames into video format.
    """

    def __init__(self, env: gymnasium.Env, name: str = "video"):
        super().__init__(env)
        self._name = name
        self._video: OrderedDict[str, List[np.ndarray]] = OrderedDict()
        self.image_keys = [k for k in self.observation_space.keys() if k != "state"]

    def get_obs_frames(self, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Retrieve stored frames as numpy arrays (filtered by keys if specified).

        Args:
            keys: Optional list of image keys to filter frames.

        Returns:
            Dict mapping image keys to numpy arrays of frames.
        """
        if keys is None:
            return {k: np.array(v) for k, v in self._video.items()}
        return {k: np.array(v) for k, v in self._video.items() if k in keys}

    def get_rendered_video(self) -> np.ndarray:
        """Combine frames into single video array (horizontal=keys, vertical=timesteps).

        Returns:
            Numpy array of combined video frames (empty if no frames).
        """
        frames: List[np.ndarray] = []
        if not self._video:
            return np.array([])

        timestep_count = len(self._video[self.image_keys[0]])
        for i in range(timestep_count):
            frame_parts = [self._video[k][i] for k in self.image_keys if k in self._video]
            frames.append(np.concatenate(frame_parts, axis=1))

        return np.concatenate(frames, axis=0)

    def _add_frame(self, obs: Dict[str, np.ndarray]) -> None:
        """Append observation frames to video storage for valid image keys.

        Args:
            obs: Observation dict containing image frames to store.
        """
        for k in self.image_keys:
            if k in obs:
                if k in self._video:
                    self._video[k].append(obs[k])
                else:
                    self._video[k] = [obs[k]]

    def reset(self, **kwargs) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset env and clear stored frames, capture initial observation frame.

        Args:
            **kwargs: Additional args for underlying env reset.

        Returns:
            (initial_obs, info) from env.
        """
        self._video.clear()
        obs, info = super().reset(**kwargs)
        self._add_frame(obs)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute step and capture resulting observation frame.

        Args:
            action: Action to execute in underlying env.

        Returns:
            (obs, reward, done, truncate, info) from env.
        """
        obs, reward, done, truncate, info = super().step(action)
        self._add_frame(obs)
        return obs, reward, done, truncate, info

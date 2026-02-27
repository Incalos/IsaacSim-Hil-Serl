import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
from agentlace.trainer import TrainerConfig

from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.vision.data_augmentations import batched_random_crop


def make_sac_pixel_agent(
    seed: int,
    sample_obs: dict,
    sample_action: torch.Tensor,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet18-pretrained",
    reward_bias: float = 0.0,
    target_entropy: Optional[float] = None,
    discount: float = 0.97,
    device: str = "cuda",
    image_size: Tuple[int, int] = (128, 128),
) -> SACAgent:
    """Create SACAgent with optimized [512, 512] ReLU architecture."""
    torch.manual_seed(seed)

    agent = SACAgent.create_pixels(
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 2,  # Optimized exploration bound
        },
        critic_network_kwargs={
            "activation": nn.ReLU(),
            "use_layer_norm": True,
            "hidden_dims": [256, 512, 512, 256, 32],
        },
        policy_network_kwargs={
            "activation": nn.ReLU(),
            "use_layer_norm": True,
            "hidden_dims": [256, 512, 512, 256, 32],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=True,  # Recommended for RLPD
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        image_size=image_size,
    )
    return agent.to(device)


def _unpack(batch: dict, image_keys: tuple) -> dict:
    """Unpack packed obs and next_obs images."""
    obs_dict = batch["observations"]
    next_obs_dict = batch.get("next_observations", {})
    for pixel_key in image_keys:
        if pixel_key in obs_dict and pixel_key not in next_obs_dict:
            obs_pixels = obs_dict[pixel_key]
            if isinstance(obs_pixels, torch.Tensor):
                # (B, T+1, ...) -> Split into t and t+1
                batch["observations"] = {**obs_dict, pixel_key: obs_pixels[:, :-1, ...]}
                batch["next_observations"] = {**next_obs_dict, pixel_key: obs_pixels[:, 1:, ...]}
                obs_dict, next_obs_dict = batch["observations"], batch["next_observations"]
    return batch


def make_batch_augmentation_func(image_keys: tuple) -> Callable:
    """Data augmentation closure with consistency between obs and next_obs."""

    def data_augmentation_fn(observations: dict, rng: torch.Generator) -> dict:
        new_obs = observations.copy()
        for pixel_key in image_keys:
            if pixel_key in new_obs:
                new_obs[pixel_key] = batched_random_crop(new_obs[pixel_key], rng=rng, padding=4, num_batch_dims=2)
        return new_obs

    def augment_batch(batch: dict, seed: int) -> dict:
        batch = _unpack(batch, image_keys)

        # FIXED: Removed 'device' from Generator.
        # torch.randint requires a CPU generator even for GPU-bound workflows.
        obs_rng = torch.Generator()
        obs_rng.manual_seed(seed)

        next_obs_rng = torch.Generator()
        next_obs_rng.manual_seed(seed + 1)

        batch["observations"] = data_augmentation_fn(batch["observations"], obs_rng)
        batch["next_observations"] = data_augmentation_fn(batch["next_observations"], next_obs_rng)
        return batch

    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589) -> TrainerConfig:
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl", description: str = "serl_launcher", debug: bool = False
) -> WandBLogger:
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({"project": project, "exp_descriptor": description, "tag": description})
    return WandBLogger(wandb_config=wandb_config, variant={}, debug=debug)

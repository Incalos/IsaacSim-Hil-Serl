import torch
import torch.nn as nn
from typing import Tuple
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
    target_entropy: float = None,
    discount: float = 0.97,
    device: str = "cuda",
    image_size: Tuple[int, int] = (128, 128),
) -> SACAgent:
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
            "std_max": 5,
        },
        critic_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        device=device,
        image_size=image_size,
    )
    return agent


def linear_schedule(step: int) -> float:
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000

    linear_step = min(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value


def _unpack(batch: dict, image_keys: tuple) -> dict:
    """
    Unpack packed obs and next_obs images.
    When pack_obs_and_next_obs=True, images are stored with an extra time dimension
    in observations only. This function splits them back into obs and next_obs.
    """
    for pixel_key in image_keys:
        # Check if pixel_key is in observations but not in next_observations
        if pixel_key in batch["observations"] and pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key]
            if isinstance(obs_pixels, torch.Tensor):
                # Packed format: (B, T+1, H, W, C) -> split into obs (B, T, H, W, C) and next_obs
                obs = dict(batch["observations"])
                next_obs = dict(batch["next_observations"])

                obs[pixel_key] = obs_pixels[:, :-1, ...]
                next_obs[pixel_key] = obs_pixels[:, 1:, ...]

                batch = dict(batch)
                batch["observations"] = obs
                batch["next_observations"] = next_obs
    return batch


def make_batch_augmentation_func(image_keys: tuple) -> callable:
    def data_augmentation_fn(observations: dict, seed: int) -> dict:
        # Create a generator from the seed
        rng = torch.Generator()
        rng.manual_seed(seed)

        for pixel_key in image_keys:
            if pixel_key in observations:
                observations = {
                    **observations,
                    pixel_key: batched_random_crop(observations[pixel_key], rng=rng, padding=4, num_batch_dims=2),
                }
        return observations

    def augment_batch(batch: dict, seed: int) -> dict:
        # First unpack packed obs and next_obs if needed
        batch = _unpack(batch, image_keys)

        obs_seed = seed
        next_obs_seed = seed + 1

        obs = data_augmentation_fn(batch["observations"], obs_seed)
        next_obs = data_augmentation_fn(batch["next_observations"], next_obs_seed)

        return {
            **batch,
            "observations": obs,
            "next_observations": next_obs,
        }

    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589) -> TrainerConfig:
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
) -> WandBLogger:
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger

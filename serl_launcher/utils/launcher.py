from typing import Tuple, Callable, Dict, Any, Optional
import torch
import torch.nn as nn
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.sac import SACAgent
from serl_launcher.vision.data_augmentations import batched_random_crop
from agentlace.trainer import TrainerConfig


def make_sac_pixel_agent(
    seed: int,
    sample_obs: dict,
    sample_action: torch.Tensor,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet18-pretrained",
    reward_bias: float = 0.0,
    target_entropy: Optional[float] = None,
    discount: float = 0.97,
    image_size: Tuple[int, int] = (128, 128),
    device: str = "cuda",
) -> SACAgent:
    """
    Create SAC (Soft Actor-Critic) agent with pixel-based observations

    Args:
        seed: Random seed for reproducibility
        sample_obs: Sample observation dictionary to infer input shapes
        sample_action: Sample action tensor to infer action space
        image_keys: Keys in observation dict corresponding to image data
        encoder_type: Type of visual encoder to use (e.g., "resnet18-pretrained")
        reward_bias: Bias added to reward values
        target_entropy: Target entropy for entropy regularization (None = auto-calculate)
        discount: Discount factor for future rewards
        image_size: Resized dimensions for input images
        device: Computation device (cuda/cpu)

    Returns:
        Initialized SACAgent with pixel-based perception
    """
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
            "activation": nn.SiLU(),
            "use_layer_norm": True,
            "hidden_dims": [256, 512, 512, 256],
        },
        policy_network_kwargs={
            "activation": nn.SiLU(),
            "use_layer_norm": True,
            "hidden_dims": [256, 512, 512, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        image_size=image_size,
        device=device,
    )
    return agent


def linear_schedule(step: int) -> float:
    """
    Linear schedule function for parameter annealing

    Args:
        step: Current training step

    Returns:
        Linearly interpolated value between init and end values
    """
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000
    linear_step = min(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value


def _unpack(batch: Dict[str, Any], image_keys: tuple) -> Dict[str, Any]:
    """
    Unpack combined observation/next_observation pixel data

    Args:
        batch: Training batch containing observations and next_observations
        image_keys: Keys for pixel data to unpack

    Returns:
        Batch with split observation/next_observation pixel data
    """
    for pixel_key in image_keys:
        if pixel_key in batch["observations"] and pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key]
            if isinstance(obs_pixels, torch.Tensor):
                obs = dict(batch["observations"])
                next_obs = dict(batch["next_observations"])
                obs[pixel_key] = obs_pixels[:, :-1, ...]
                next_obs[pixel_key] = obs_pixels[:, 1:, ...]
                batch = dict(batch)
                batch["observations"] = obs
                batch["next_observations"] = next_obs
    return batch


def make_batch_augmentation_func(image_keys: tuple) -> Callable[[Dict[str, Any], int], Dict[str, Any]]:
    """
    Create batch augmentation function for pixel observations

    Args:
        image_keys: Keys for pixel data to augment

    Returns:
        Augmentation function that processes entire batches
    """

    def data_augmentation_fn(observations: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Apply random crop augmentation to pixel observations"""
        rng = torch.Generator()
        rng.manual_seed(seed)
        for pixel_key in image_keys:
            if pixel_key in observations:
                observations = {
                    **observations,
                    pixel_key: batched_random_crop(observations[pixel_key], rng=rng, padding=4, num_batch_dims=2),
                }
        return observations

    def augment_batch(batch: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Unpack batch and apply augmentation to both observations and next_observations"""
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
    """
    Create default TrainerConfig with specified port configuration

    Args:
        port_number: Main communication port
        broadcast_port: Broadcast port for trainer communication

    Returns:
        Initialized TrainerConfig object
    """
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
    """
    Create WandBLogger instance with default configuration

    Args:
        project: WandB project name
        description: Experiment description/tag
        debug: If True, run in debug mode (offline logging)

    Returns:
        Initialized WandBLogger object
    """
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

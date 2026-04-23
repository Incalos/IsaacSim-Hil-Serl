import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.append(_project_root)

import glob
import pickle as pkl
import random
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from absl import app, flags
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 300, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")
flags.DEFINE_string("checkpoint_name", "checkpoint.pth", "Name of the model checkpoint file.")
flags.DEFINE_string("data_path", "classifier_data", "Path to save the data files.")


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    rng: torch.Generator,
    device: torch.device,
    data_augmentation_fn: callable,
) -> Tuple[float, float]:
    """Execute single training step for reward classifier.

    Args:
        model: Reward classifier model
        optimizer: Optimizer for model parameters
        batch: Training batch containing observations and labels
        rng: Random number generator for augmentation
        device: Compute device (CPU/GPU)
        data_augmentation_fn: Function to apply data augmentation on observations

    Returns:
        Tuple of training loss value and accuracy value
    """
    model.train()
    optimizer.zero_grad()

    if isinstance(batch["observations"], dict):
        batch["observations"] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["observations"].items()
        }
    batch["labels"] = batch["labels"].to(device)

    obs = data_augmentation_fn(rng, batch["observations"])
    logits = model(obs, train=True)
    labels = batch["labels"].float().view(-1, 1)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (preds == labels).float().mean()

    return loss.item(), accuracy.item()


def main(_: Any) -> None:
    """Main function to train reward classifier for success/failure classification.

    Args:
        _: Unused argument from absl app
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, classifier=False)
    data_path = os.path.join(os.path.dirname(__file__), "experiments", FLAGS.exp_name, FLAGS.data_path)
    robot_params = config.robot_params

    pos_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=40000, include_label=True, device="cpu")
    success_paths = glob.glob(os.path.join(data_path, "*success*.pkl"))
    for path in success_paths:
        with open(path, "rb") as f:
            success_data = pkl.load(f)
        for trans in success_data:
            if "images" in trans["observations"].keys():
                continue
            trans["labels"] = 1
            if "actions" not in trans:
                trans["actions"] = env.action_space.sample()
            pos_buffer.insert(trans)
    pos_iterator = pos_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size // 2}, device="cpu")

    neg_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=40000, include_label=True, device="cpu")
    max_samples_per_file = 100
    failure_paths = glob.glob(os.path.join(data_path, "*failure*.pkl"))
    for path in failure_paths:
        with open(path, "rb") as f:
            failure_data = pkl.load(f)
        if len(failure_data) > max_samples_per_file:
            sampled_indices = random.sample(range(len(failure_data)), max_samples_per_file)
        else:
            sampled_indices = range(len(failure_data))
        for i in sampled_indices:
            trans = failure_data[i]
            if "images" in trans["observations"].keys():
                continue
            trans["labels"] = 0
            if "actions" not in trans:
                trans["actions"] = env.action_space.sample()
            neg_buffer.insert(trans)
    neg_iterator = neg_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size // 2}, device="cpu")

    print(f"Failed buffer size: {len(neg_buffer)}")
    print(f"Success buffer size: {len(pos_buffer)}")

    model = create_classifier(image_keys=robot_params["classifier_keys"], n_way=2, img_size=robot_params["image_size"])
    model.to(device)
    optimizer = model.optimizer
    rng = torch.Generator()
    rng.manual_seed(42)

    def data_augmentation_fn(rng: torch.Generator, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply batched random crop augmentation to image observations.

        Args:
            rng: Random number generator
            observations: Dictionary of observation tensors

        Returns:
            Augmented observations dictionary
        """
        for pixel_key in robot_params["classifier_keys"]:
            observations[pixel_key] = batched_random_crop(observations[pixel_key], rng, padding=4, num_batch_dims=2)
        return observations

    for epoch in tqdm(range(FLAGS.num_epochs)):
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        batch = concat_batches(pos_sample, neg_sample, axis=0)

        train_loss, train_accuracy = train_step(model, optimizer, batch, rng, device, data_augmentation_fn)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    ckpt_dir = os.path.join(os.path.dirname(data_path), "classifier_ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, FLAGS.checkpoint_name)
    torch.save(
        {
            "epoch": FLAGS.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
        },
        save_path,
    )
    print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    app.run(main)

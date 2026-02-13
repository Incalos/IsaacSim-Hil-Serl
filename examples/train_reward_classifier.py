import os
import sys

# Ensure project modules are discoverable by resolving the root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import glob
import pickle as pkl
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from absl import app, flags
import random
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 300, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # Environment instance is used here primarily to infer the observation and action space shapes
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    data_path = os.path.join(os.path.dirname(__file__), "experiments", FLAGS.exp_name, "classifier_data")
    # Initialize ReplayBuffer for Positive (Success) samples with binary label support
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
    # Initialize ReplayBuffer for Negative (Failure) samples
    neg_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=40000, include_label=True, device="cpu")
    max_samples_per_file = 100
    failure_paths = glob.glob(os.path.join(data_path, "*failure*.pkl"))
    for path in failure_paths:
        with open(path, "rb") as f:
            failure_data = pkl.load(f)
        # Limit negative samples per file to prevent single long trajectories from dominating the buffer
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
    # Create the binary classifier using the experiment's specific image keys and resolution
    model = create_classifier(image_keys=config.classifier_keys, n_way=2, img_size=config.image_size)
    model.to(device)
    optimizer = model.optimizer
    rng = torch.Generator()
    rng.manual_seed(42)

    def data_augmentation_fn(rng, observations):
        # Apply visual shifts to make the classifier robust to slight camera perturbations
        for pixel_key in config.classifier_keys:
            observations[pixel_key] = batched_random_crop(observations[pixel_key], rng, padding=4, num_batch_dims=2)
        return observations

    def train_step(model, optimizer, batch, rng):
        model.train()
        optimizer.zero_grad()
        # Transfer the observation dictionary and labels to the active compute device (GPU)
        if isinstance(batch["observations"], dict):
            batch["observations"] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["observations"].items()
            }
        batch["labels"] = batch["labels"].to(device)
        obs = data_augmentation_fn(rng, batch["observations"])
        logits = model(obs, train=True)
        labels = batch["labels"].float().view(-1, 1)
        # Use Binary Cross Entropy with Logits for numerical stability (replaces Sigmoid + BCE)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            accuracy = (preds == labels).float().mean()
        return loss.item(), accuracy.item()

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Balanced sampling: 50% success and 50% failure transitions per batch
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        batch = concat_batches(pos_sample, neg_sample, axis=0)
        train_loss, train_accuracy = train_step(model, optimizer, batch, rng)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    ckpt_dir = os.path.join(os.path.dirname(data_path), "classifier_ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, "classifier_model.pth")
    # Save the full state dictionary for later evaluation or resumed training
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

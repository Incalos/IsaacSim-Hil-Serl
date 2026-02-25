import pickle as pkl
import numpy as np
from tqdm import tqdm
from absl import app, flags
from experiments.mappings import CONFIG_MAPPING
import cv2

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("demo_file", None, "Path to the demo pickle file to replay.")
flags.DEFINE_integer("max_episodes", None, "Maximum number of episodes to replay (None for all).")


def _extract_obs_images(obs_dict):
    """Extract image-like arrays from an observation dict and convert to uint8 HxWx3."""
    images = []
    if not isinstance(obs_dict, dict):
        return images

    for _, v in obs_dict.items():

        v = np.squeeze(v)

        if not isinstance(v, np.ndarray):
            continue
        if v.ndim != 3:
            continue

        img = v
        # Handle CHW -> HWC
        if img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))

        # Require 1 or 3 channels at the end
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] != 3:
            continue

        # Normalize/convert to uint8
        if img.dtype != np.uint8:
            img_min, img_max = float(img.min()), float(img.max())
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        images.append(img)

    return images


def _show_stacked_obs_images(obs_dict, window_name="demo_observations"):
    """Stack all image observations horizontally and show them with OpenCV."""
    images = _extract_obs_images(obs_dict)
    if not images:
        return

    # Resize all images to the same height for clean concatenation
    target_h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = int(w * (target_h / float(h)))
            img = cv2.resize(img, (new_w, target_h))
        resized.append(img)

    stacked = np.concatenate(resized, axis=1)
    cv2.imshow(window_name, stacked)
    # Small delay to refresh the window; 1 ms is enough for replay
    cv2.waitKey(1)


def replay_demos(demo_file, exp_name, max_episodes=None):
    """
    Replay demonstrations by loading actions from pickle file and executing them in the environment.

    Args:
        demo_file: Path to the pickle file containing transitions
        exp_name: Experiment name for loading the correct environment config
        max_episodes: Maximum number of episodes to replay (None for all)
    """
    # Load the demo file
    print(f"Loading demo file: {demo_file}")
    with open(demo_file, "rb") as f:
        transitions = pkl.load(f)

    print(f"Loaded {len(transitions)} transitions")

    # Validate and load the experiment configuration
    assert exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[exp_name]()

    # Initialize environment (same as record_demos.py)
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    # Split transitions into episodes based on 'dones' flag
    episodes = []
    current_episode = []

    for transition in transitions:
        current_episode.append(transition)
        if transition.get("dones", False):
            episodes.append(current_episode)
            current_episode = []

    # Add the last episode if it doesn't end with done
    if current_episode:
        episodes.append(current_episode)

    print(f"Found {len(episodes)} episodes in the demo file")

    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        print(f"Replaying first {len(episodes)} episodes")

    # Replay each episode
    success_count = 0
    total_reward = 0

    for episode_idx, episode in enumerate(tqdm(episodes, desc="Replaying episodes")):
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0

        # Execute actions from the episode
        for step_idx, transition in enumerate(episode):
            # Visualize stacked observation images from the recorded transition
            if "observations" in transition:
                _show_stacked_obs_images(transition["observations"])

            # Extract action from the transition
            action = transition["actions"]

            # Ensure action is numpy array with correct shape
            if isinstance(action, np.ndarray):
                action = action.copy()
                # Remove batch dimension if present (shape like (1, 6) -> (6,))
                if len(action.shape) > 1 and action.shape[0] == 1:
                    action = action.squeeze(0)
            else:
                action = np.array(action)

            # Ensure action has correct dtype (float64 -> float32 if needed)
            if action.dtype != np.float32 and action.dtype != np.float64:
                action = action.astype(np.float64)

            # Step the environment with the recorded action
            try:
                _, rew, done, truncated, info = env.step(action)
                episode_reward += rew
            except Exception as e:
                print(f"\nError executing action at episode {episode_idx}, step {step_idx}: {e}")
                print(f"Action shape: {action.shape}, dtype: {action.dtype}")
                raise

            # Check if episode ended early (should match the recorded done)
            if done or truncated:
                # Verify if this matches the recorded done state
                recorded_done = transition.get("dones", False)
                if recorded_done:
                    break
                else:
                    print(
                        f"\nWarning: Episode {episode_idx} ended early at step {step_idx}, but transition indicates not done"
                    )
                    break

        # Check success
        if info.get("succeed", False):
            success_count += 1

        total_reward += episode_reward
        print(
            f"\nEpisode {episode_idx + 1}/{len(episodes)}: Reward={episode_reward:.2f}, Success={info.get('succeed', False)}"
        )

    # Print summary
    print("\n" + "=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)
    print(f"Total episodes replayed: {len(episodes)}")
    print(f"Successful episodes: {success_count}/{len(episodes)}")
    print(f"Success rate: {success_count / len(episodes) * 100:.2f}%")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per episode: {total_reward / len(episodes):.2f}")

    # Close any OpenCV windows that were opened
    cv2.destroyAllWindows()


def main(_):
    if FLAGS.demo_file is None or FLAGS.exp_name is None:
        print("Error: Both --demo_file and --exp_name must be specified")
        print("Usage: python replay_demos.py --exp_name=so101_pick_oranges --demo_file=path/to/demo.pkl")
        return

    replay_demos(demo_file=FLAGS.demo_file, exp_name=FLAGS.exp_name, max_episodes=FLAGS.max_episodes)


if __name__ == "__main__":
    app.run(main)

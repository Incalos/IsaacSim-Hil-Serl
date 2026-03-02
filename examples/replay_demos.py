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
    # Extract and format image arrays from observation dictionary
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
        # Convert CHW format to HWC if needed
        if img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))

        # Ensure 3-channel output (convert 1-channel to 3-channel)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] != 3:
            continue

        # Normalize and convert to uint8 format for OpenCV display
        if img.dtype != np.uint8:
            img_min, img_max = float(img.min()), float(img.max())
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        images.append(img)

    return images


def _show_stacked_obs_images(obs_dict, window_name="demo_observations"):
    # Stack observation images horizontally and display with OpenCV
    images = _extract_obs_images(obs_dict)
    if not images:
        return

    # Resize all images to same height for consistent concatenation
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
    cv2.waitKey(1)


def replay_demos(demo_file, exp_name, max_episodes=None):
    # Load demonstration data from pickle file
    print(f"Loading demo file: {demo_file}")
    with open(demo_file, "rb") as f:
        transitions = pkl.load(f)

    print(f"Loaded {len(transitions)} transitions")

    # Load experiment configuration and validate
    assert exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[exp_name]()

    # Initialize environment with real execution (non-fake)
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    # Split transitions into episodes using 'dones' flag
    episodes = []
    current_episode = []
    for transition in transitions:
        current_episode.append(transition)
        if transition.get("dones", False):
            episodes.append(current_episode)
            current_episode = []

    # Add remaining transitions as final episode if not terminated
    if current_episode:
        episodes.append(current_episode)

    print(f"Found {len(episodes)} episodes in the demo file")

    # Limit episodes if max_episodes is specified
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        print(f"Replaying first {len(episodes)} episodes")

    # Initialize metrics tracking
    success_count = 0
    total_reward = 0

    # Replay each episode sequentially
    for episode_idx, episode in enumerate(tqdm(episodes, desc="Replaying episodes")):
        env.reset()

        # Execute each step in the episode
        for step_idx, transition in enumerate(episode):
            # Visualize recorded observations
            if "observations" in transition:
                _show_stacked_obs_images(transition["observations"])

            # Extract and format action from transition
            action = transition["actions"]
            if isinstance(action, np.ndarray):
                action = action.copy()
                # Remove batch dimension if present
                if len(action.shape) > 1 and action.shape[0] == 1:
                    action = action.squeeze(0)
            else:
                action = np.array(action)

            # Ensure correct numeric type for action
            if action.dtype != np.float32 and action.dtype != np.float64:
                action = action.astype(np.float64)

            # Execute action in environment
            try:
                _, rew, done, truncated, info = env.step(action)
                total_reward += rew
            except Exception as e:
                print(f"\nError executing action at episode {episode_idx}, step {step_idx}: {e}")
                print(f"Action shape: {action.shape}, dtype: {action.dtype}")
                raise

            # Check for early episode termination
            if done or truncated:
                recorded_done = transition.get("dones", False)
                if recorded_done:
                    break
                else:
                    print(f"\nWarning: Episode {episode_idx} ended early at step {step_idx}, but transition indicates not done")
                    break

        # Track successful episodes
        if info.get("succeed", False):
            success_count += 1

        print(f"\nEpisode {episode_idx + 1}/{len(episodes)}: Success={info.get('succeed', False)}")

    # Print replay summary statistics
    print("\n" + "=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)
    print(f"Total episodes replayed: {len(episodes)}")
    print(f"Successful episodes: {success_count}/{len(episodes)}")
    print(f"Success rate: {success_count / len(episodes) * 100:.2f}%")
    print(f"Average reward per episode: {total_reward / len(episodes):.2f}")

    # Clean up OpenCV windows
    cv2.destroyAllWindows()


def main(_):
    # Validate required command line arguments
    if FLAGS.demo_file is None or FLAGS.exp_name is None:
        print("Error: Both --demo_file and --exp_name must be specified")
        return

    # Start demo replay process
    replay_demos(demo_file=FLAGS.demo_file, exp_name=FLAGS.exp_name, max_episodes=FLAGS.max_episodes)


if __name__ == "__main__":
    app.run(main)

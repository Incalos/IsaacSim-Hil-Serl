import pickle as pkl
import numpy as np
import cv2
import os
from typing import Dict, List, Any
from tqdm import tqdm
from absl import app, flags
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("demo_file", None, "Path to the demo pickle file to replay.")
flags.DEFINE_integer("max_episodes", None, "Maximum number of episodes to replay (None for all).")


def _extract_obs_images(obs_dict: Dict[str, Any]) -> List[np.ndarray]:
    """Extract and format image arrays from observation dictionary.
    
    Args:
        obs_dict: Observation dictionary containing image tensors/arrays
        
    Returns:
        List of formatted image arrays (uint8, HWC, RGB)
    """
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
        if img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] != 3:
            continue

        if img.dtype != np.uint8:
            img_min, img_max = float(img.min()), float(img.max())
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        images.append(img)

    return images


def _show_stacked_obs_images(obs_dict: Dict[str, Any], window_name: str = "demo_observations") -> None:
    """Stack observation images horizontally and display with OpenCV.
    
    Args:
        obs_dict: Observation dictionary containing image data
        window_name: Name of OpenCV display window
    """
    images = _extract_obs_images(obs_dict)
    if not images:
        return

    target_h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = int(w * (target_h / float(h)))
            img = cv2.resize(img, (new_w, target_h))
        resized.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    stacked = np.concatenate(resized, axis=1)
    cv2.imshow(window_name, stacked)
    cv2.waitKey(1)


def replay_demos() -> None:
    """Load and replay demonstration data, track success rate and reward metrics."""
    demo_file = os.path.join(os.path.dirname(__file__), "experiments", FLAGS.exp_name, FLAGS.demo_file)
    print(f"Loading demo file: {demo_file}")
    with open(demo_file, "rb") as f:
        transitions = pkl.load(f)

    print(f"Loaded {len(transitions)} transitions")

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, classifier=True)

    episodes = []
    current_episode = []
    for transition in transitions:
        current_episode.append(transition)
        if transition.get("dones", False):
            episodes.append(current_episode)
            current_episode = []

    if current_episode:
        episodes.append(current_episode)

    print(f"Found {len(episodes)} episodes in the demo file")

    if FLAGS.max_episodes is not None:
        episodes = episodes[:FLAGS.max_episodes]
        print(f"Replaying first {len(episodes)} episodes")

    success_count = 0
    total_reward = 0

    for episode_idx, episode in enumerate(tqdm(episodes, desc="Replaying episodes")):
        env.reset()

        for step_idx, transition in enumerate(episode):
            if "observations" in transition:
                _show_stacked_obs_images(transition["observations"])

            action = transition["actions"]
            if isinstance(action, np.ndarray):
                action = action.copy()
                if len(action.shape) > 1 and action.shape[0] == 1:
                    action = action.squeeze(0)
            else:
                action = np.array(action)

            if action.dtype != np.float32 and action.dtype != np.float64:
                action = action.astype(np.float64)

            try:
                _, rew, done, truncated, info = env.step(action)
                total_reward += rew
            except Exception as e:
                print(f"\nError executing action at episode {episode_idx}, step {step_idx}: {e}")
                print(f"Action shape: {action.shape}, dtype: {action.dtype}")
                raise

            if done or truncated:
                recorded_done = transition.get("dones", False)
                if recorded_done:
                    break
                else:
                    print(f"\nWarning: Episode {episode_idx} ended early at step {step_idx}, but transition indicates not done")
                    break

        if info.get("succeed", False):
            success_count += 1

        print(f"\nEpisode {episode_idx + 1}/{len(episodes)}: Success={info.get('succeed', False)}")

    print("\n" + "=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)
    print(f"Total episodes replayed: {len(episodes)}")
    print(f"Successful episodes: {success_count}/{len(episodes)}")
    print(f"Success rate: {success_count / len(episodes) * 100:.2f}%")
    print(f"Average reward per episode: {total_reward / len(episodes):.2f}")

    cv2.destroyAllWindows()


def main(_: Any) -> None:
    """Main function to validate arguments and start demo replay.
    
    Args:
        _: Unused argument from absl app
    """
    if FLAGS.demo_file is None or FLAGS.exp_name is None:
        print("Error: Both --demo_file and --exp_name must be specified")
        return

    replay_demos()


if __name__ == "__main__":
    app.run(main)

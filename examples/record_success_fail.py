import copy
import os
import pickle as pkl
import numpy as np
from typing import Dict, List, Any
import datetime
from tqdm import tqdm
from absl import app, flags
from pynput import keyboard
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 100, "Number of successful transitions to collect.")
flags.DEFINE_integer("save_interval", 5, "Frequency of saving success batches to disk.")
flags.DEFINE_string("data_path", "classifier_data", "Path to save the data files.")

success_key = False
recording_started = False


def on_press(key: keyboard.Key) -> None:
    """Keyboard press handler for recording control.
    
    Args:
        key: Pressed keyboard key
    """
    global success_key, recording_started
    try:
        if key == keyboard.Key.space:
            if recording_started:
                success_key = True
        elif hasattr(key, "char") and key.char == "b":
            recording_started = not recording_started
            status = "STARTED (recording enabled)" if recording_started else "PAUSED (recording disabled)"
            print(f"\n[Status: {status}]")
    except AttributeError:
        pass


def save_pkl(path: str, filename: str, data: List[Dict[str, Any]]) -> None:
    """Save data to pickle file.
    
    Args:
        path: Directory path for saving file
        filename: Name of the pickle file
        data: Data to save (list of transitions)
    """
    if not data:
        return
    full_path = os.path.join(path, filename)
    with open(full_path, "wb") as f:
        pkl.dump(data, f)


def main(_: Any) -> None:
    """Main function to record success/failure transitions with keyboard control.
    
    Args:
        _: Unused argument from absl app
    """
    global success_key, recording_started
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, classifier=False)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "experiments", FLAGS.exp_name, FLAGS.data_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    success_buffer = []
    failure_buffer = []
    episode_idx = 0
    success_batch_idx = 0
    success_total = 0
    pbar = tqdm(total=FLAGS.successes_needed, desc="Success Progress")
    obs, _ = env.reset()

    print("--- Controls: [b] Toggle Record | [Space] Mark Success & Reset | [Ctrl+C] Exit ---")
    print("[Info] Press 'b' to start recording and enable teleoperation.")

    try:
        while success_total < FLAGS.successes_needed:
            if not recording_started:
                continue

            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)

            if "intervene_action" in info:
                actions = info["intervene_action"]

            transition = copy.deepcopy(dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - (done or truncated),
                dones=done,
            ))
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if success_key:
                save_pkl(save_path, f"{FLAGS.exp_name}_failure_ep_{episode_idx}_{uuid}.pkl", failure_buffer)

                success_buffer.append(transition)
                success_total += 1
                pbar.update(1)

                if len(success_buffer) >= FLAGS.save_interval:
                    save_pkl(save_path, f"{FLAGS.exp_name}_success_batch_{success_batch_idx}_{uuid}.pkl", success_buffer)
                    success_buffer = []
                    success_batch_idx += 1

                failure_buffer = []
                success_key = False
                recording_started = False
                episode_idx += 1
                obs, _ = env.reset()
                print(f"\n[SUCCESS] Ep {episode_idx} saved. Recording paused. Press 'b' to restart.")
                continue

            failure_buffer.append(transition)
            obs = next_obs

            if done or truncated:
                save_pkl(save_path, f"{FLAGS.exp_name}_failure_ep_{episode_idx}_{uuid}.pkl", failure_buffer)
                failure_buffer = []
                recording_started = False
                episode_idx += 1
                obs, _ = env.reset()
                print(f"\n[DONE] Ep {episode_idx} saved. Recording paused. Press 'b' to restart.")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")

    finally:
        print(f"All data saved to: {save_path}")


if __name__ == "__main__":
    app.run(main)

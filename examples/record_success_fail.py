import copy
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from absl import app, flags
from pynput import keyboard
import datetime
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")
flags.DEFINE_integer("save_interval", 10, "Frequency of saving success batches to disk.")
flags.DEFINE_string("save_path", None, "Path to save the output files.")

# Global state flags for keyboard synchronization
success_key = False
recording_started = False


def on_press(key):
    global success_key, recording_started
    try:
        if key == keyboard.Key.space:
            success_key = True
        elif hasattr(key, "char") and key.char == "b":
            recording_started = not recording_started
            status = "STARTED" if recording_started else "PAUSED"
            print(f"\n[Status: {status}]")
    except AttributeError:
        pass


def save_pkl(path, filename, data):
    if not data:
        return
    full_path = os.path.join(path, filename)
    with open(full_path, "wb") as f:
        pkl.dump(data, f)


def main(_):
    global success_key, recording_started
    # Initialize keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Validate experiment configuration
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    # Set up save directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "experiments", FLAGS.exp_name, FLAGS.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize data buffers and counters
    success_buffer = []
    failure_buffer = []
    episode_idx = 0
    success_batch_idx = 0
    success_total = 0
    pbar = tqdm(total=FLAGS.successes_needed, desc="Success Progress")
    obs, _ = env.reset()

    # Print control instructions
    print("--- Controls: [b] Toggle Record | [Space] Mark Success & Reset | [Ctrl+C] Exit ---")

    try:
        while success_total < FLAGS.successes_needed:
            if not recording_started:
                continue

            # Generate action and step environment
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)

            # Override action with teleoperation input if available
            if "intervene_action" in info:
                actions = info["intervene_action"]

            # Create transition dictionary
            transition = copy.deepcopy(dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - (done or truncated),
                dones=done,
            ))
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Handle success signal from user
            if success_key:
                # Save failure buffer as negative samples
                save_pkl(save_path, f"{FLAGS.exp_name}_failure_ep_{episode_idx}_{uuid}.pkl", failure_buffer)

                # Add successful transition to buffer
                success_buffer.append(transition)
                success_total += 1
                pbar.update(1)

                # Save success batch if interval is reached
                if len(success_buffer) >= FLAGS.save_interval:
                    save_pkl(save_path, f"{FLAGS.exp_name}_success_batch_{success_batch_idx}_{uuid}.pkl", success_buffer)
                    success_buffer = []
                    success_batch_idx += 1

                # Reset buffers and state flags
                failure_buffer = []
                success_key = False
                recording_started = False
                episode_idx += 1
                obs, _ = env.reset()
                print(f"\n[SUCCESS] Ep {episode_idx} saved. Recording paused.")
                continue

            # Add transition to failure buffer for negative samples
            failure_buffer.append(transition)
            obs = next_obs

            # Handle natural episode termination
            if done or truncated:
                save_pkl(save_path, f"{FLAGS.exp_name}_failure_ep_{episode_idx}_{uuid}.pkl", failure_buffer)
                failure_buffer = []
                recording_started = False
                episode_idx += 1
                obs, _ = env.reset()
                print(f"\n[DONE] Ep {episode_idx} saved. Recording paused.")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")

    finally:
        print(f"All data saved to: {save_path}")


if __name__ == "__main__":
    app.run(main)

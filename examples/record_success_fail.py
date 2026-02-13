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
# Global state flags for keyboard synchronization
success_key = False
recording_started = False


def on_press(key):
    global success_key, recording_started
    try:
        # Spacebar marks the current transition as a 'Success'
        if key == keyboard.Key.space:
            success_key = True
        # 'b' toggles the recording state to prevent junk data during setup
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
    # Initialize asynchronous keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # Environment is initialized without a classifier as we are currently collecting the data to train one
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "experiments", FLAGS.exp_name, "classifier_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # failure_buffer stores intermediate steps; success_buffer stores the goal-reaching transitions
    success_buffer = []
    failure_buffer = []
    episode_idx = 0
    success_batch_idx = 0
    success_total = 0
    pbar = tqdm(total=FLAGS.successes_needed, desc="Success Progress")
    obs, _ = env.reset()
    print("--- Controls: [b] Toggle Record | [Space] Mark Success & Reset | [Ctrl+C] Exit ---")
    try:
        while success_total < FLAGS.successes_needed:
            if not recording_started:
                continue
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)
            # Prioritize manual intervention actions (teleoperation)
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - (done or truncated),
                    dones=done,
                )
            )
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Case A: User signals success
            if success_key:
                # Save all preceding steps in this episode as negative (failure) samples
                save_pkl(save_path, f"{FLAGS.exp_name}_failure_ep_{episode_idx}_{uuid}.pkl", failure_buffer)
                # Store the specific transition that achieved the goal
                success_buffer.append(transition)
                success_total += 1
                pbar.update(1)
                # Periodically serialize success batches to avoid memory overflow or data loss
                if len(success_buffer) >= FLAGS.save_interval:
                    save_pkl(
                        save_path, f"{FLAGS.exp_name}_success_batch_{success_batch_idx}_{uuid}.pkl", success_buffer
                    )
                    success_buffer = []
                    success_batch_idx += 1
                failure_buffer = []
                success_key = False
                recording_started = False
                episode_idx += 1
                obs, _ = env.reset()
                print(f"\n[SUCCESS] Ep {episode_idx} saved. Recording paused.")
                continue
            # Case B: Standard operation (collecting negative samples)
            failure_buffer.append(transition)
            obs = next_obs
            # Case C: Natural episode termination without reaching success
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

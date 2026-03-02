import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

# Global flag to control recording state
recording_started = False


def on_press(key):
    global recording_started
    try:
        if hasattr(key, "char") and key.char == "b":
            recording_started = not recording_started
            status = "STARTED" if recording_started else "PAUSED"
            print(f"\n[Recording: {status}]")
    except AttributeError:
        pass


def main(_):
    global recording_started
    # Initialize keyboard listener for recording control
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Press 'b' to start/pause recording trajectory")

    # Validate experiment config and initialize environment
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    obs, info = env.reset()
    print("Reset done")

    # Initialize data storage and progress tracking
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []

    # Main collection loop until target successful demos are collected
    while success_count < success_needed:
        # Initialize action with zero values (overwritten by human intervention)
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, _, info = env.step(actions)

        # Override action with human intervention input if available
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # Build transition dict for RL dataset
        transition = copy.deepcopy(dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=rew,
            masks=1.0 - done,
            dones=done,
            infos=info,
        ))

        # Append transition to current trajectory if recording is active
        if recording_started:
            trajectory.append(transition)

        # Update progress bar with current recording state
        pbar.set_description(f"Recording: {'ON' if recording_started else 'OFF'}")
        obs = next_obs

        # Handle episode termination
        if done:
            was_recording = recording_started

            # Auto-stop recording when episode ends
            if recording_started:
                recording_started = False
                print(f"\n[AUTO-STOPPED] Episode ended. Press 'b' to start next recording.")

            # Save trajectory only if episode was successful and recording was active
            if info["succeed"] and was_recording:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)

            # Reset trajectory buffer and environment for next attempt
            trajectory = []
            obs, info = env.reset()
            print("Reset done. Press 'b' to start recording next trajectory.")

    # Create save directory if not exists
    save_path = os.path.join(os.path.dirname(__file__), "experiments", FLAGS.exp_name, "demo_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate unique filename with timestamp to avoid overwrites
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_path}/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"

    # Save collected demonstrations to pickle file
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")


if __name__ == "__main__":
    app.run(main)

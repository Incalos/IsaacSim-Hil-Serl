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

# Global state flag for keyboard synchronization
recording_started = False


def on_press(key):
    global recording_started
    try:
        # 'b' toggles the recording state to start recording trajectory
        if hasattr(key, "char") and key.char == "b":
            recording_started = not recording_started
            status = "STARTED" if recording_started else "PAUSED"
            print(f"\n[Recording: {status}]")
    except AttributeError:
        pass


def main(_):
    global recording_started
    # Initialize asynchronous keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Press 'b' to start/pause recording trajectory")
    # Validate and load the experiment configuration from the central registry
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # Environment is initialized with the classifier enabled to automatically detect task success
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    while success_count < success_needed:
        # Action placeholder; usually overwritten by human teleoperation via the intervention wrapper
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        # Capture the actual control input if a human operator intervened during the step
        if "intervene_action" in info:
            actions = info["intervene_action"]
        # Construct transition dictionary for offline reinforcement learning datasets
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        # Only append to trajectory if recording has been started (b key pressed)
        if recording_started:
            trajectory.append(transition)
        pbar.set_description(f"Return: {returns} | Recording: {'ON' if recording_started else 'OFF'}")
        obs = next_obs
        if done:
            # Check if we were recording before stopping
            was_recording = recording_started
            # Automatically stop recording when episode ends
            if recording_started:
                recording_started = False
                print(f"\n[Recording: AUTO-STOPPED] Episode ended. Press 'b' to start next recording.")
            # Data Filtering: Only commit the trajectory to the final dataset if the task was successful and was being recorded
            if info["succeed"] and was_recording:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            # Reset buffers and environment for the next demonstration attempt
            trajectory = []
            returns = 0
            obs, info = env.reset()
            print("Reset done. Press 'b' to start recording next trajectory.")
    # Path construction for the persistent demonstration storage
    save_path = os.path.join(os.path.dirname(__file__), "experiments", FLAGS.exp_name, "demo_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Generate a unique timestamped filename to prevent overwriting previous collections
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_path}/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")


if __name__ == "__main__":
    app.run(main)

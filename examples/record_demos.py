import os
import datetime
import copy
import pickle as pkl
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from absl import app, flags
from pynput import keyboard
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")


class DemoRecorder:
    """Demo recorder class encapsulates all recording-related states and logic"""

    def __init__(self, exp_name: str, successes_needed: int):
        self.exp_name = exp_name
        self.successes_needed = successes_needed
        self.recording_started = False  # Flag to indicate if recording is active
        self.env = None  # Environment instance
        self.current_obs = None  # Current observation from environment
        self.current_info = None  # Current info dict from environment
        self.trajectory: List[Dict[str, Any]] = []  # Cache for current episode trajectory
        self.all_transitions: List[Dict[str, Any]] = []  # All successful trajectory transitions
        self.success_count = 0  # Count of successfully recorded demos
        self.pbar = None  # Progress bar instance
        self.key_listener = None  # Keyboard listener instance

    def init_environment(self):
        # Initialize experiment environment based on exp_name
        assert self.exp_name in CONFIG_MAPPING, f"Experiment {self.exp_name} not found in CONFIG_MAPPING"
        config = CONFIG_MAPPING[self.exp_name]()
        self.env = config.get_environment(fake_env=False, classifier=True)
        self.current_obs, self.current_info = self.env.reset()
        print("✅ Environment reset completed")

    def init_keyboard_listener(self):
        # Initialize keyboard listener to control recording start/pause (press 'b' to toggle)
        def on_press(key):
            try:
                if hasattr(key, "char") and key.char == "b":
                    self.recording_started = not self.recording_started
                    status = "STARTED" if self.recording_started else "PAUSED"
                    print(f"\n[Recording: {status}]")
                    if self.pbar:
                        self.pbar.set_description(f"Recording: {'ON' if self.recording_started else 'OFF'}")
                        self.pbar.refresh()
            except AttributeError:
                pass

        self.key_listener = keyboard.Listener(on_press=on_press)
        self.key_listener.start()
        print("⌨️ Keyboard listener started (press 'b' to start/pause recording)")

    def init_progress_bar(self):
        # Initialize progress bar to track successful demo collection
        self.pbar = tqdm(total=self.successes_needed, desc=f"Recording: {'ON' if self.recording_started else 'OFF'}", ncols=80, leave=True)

    def process_env_step(self):
        # Process single environment step (only when recording is active)
        action_shape = self.env.action_space.sample().shape
        actions = np.zeros(action_shape)

        # Execute environment step with initial zero action
        next_obs, rew, done, _, info = self.env.step(actions)

        # Override action with human intervention action if available
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # Construct transition data for current step
        transition = copy.deepcopy({
            "observations": self.current_obs,
            "actions": actions,
            "next_observations": next_obs,
            "rewards": rew,
            "masks": 1.0 - done,
            "dones": done,
            "infos": info,
        })

        # Append transition to current trajectory and update observation
        self.trajectory.append(transition)
        self.current_obs = next_obs

        # Handle episode termination if done
        if done:
            self.handle_episode_end(info, done)

    def handle_episode_end(self, info: Dict[str, Any], done: bool):
        # Handle logic when episode ends
        was_recording = self.recording_started

        # Auto-stop recording when episode finishes
        if self.recording_started:
            self.recording_started = False
            print("\n[Auto-stopped] Episode finished. Press 'b' to start next recording.")
            self.pbar.set_description(f"Recording: OFF")
            self.pbar.refresh()

        # Save trajectory if episode was successful and recording was active
        if info.get("succeed", False) and was_recording:
            self.all_transitions.extend(copy.deepcopy(self.trajectory))
            self.success_count += 1
            self.pbar.update(1)

        # Reset trajectory and environment for next episode
        self.trajectory = []
        self.current_obs, self.current_info = self.env.reset()
        print("✅ Environment reset completed. Press 'b' to start next recording.")

    def save_demos(self):
        # Save recorded demo transitions to pickle file
        save_dir = os.path.join(os.path.dirname(__file__), "experiments", self.exp_name, "demo_data")
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(save_dir, f"{self.exp_name}_{self.successes_needed}_demos_{timestamp}.pkl")

        # Write data to file
        with open(file_path, "wb") as f:
            pkl.dump(self.all_transitions, f)

        print(f"\n🎉 Successfully saved {self.successes_needed} demos to:")
        print(f"   {file_path}")

    def run(self):
        # Main execution flow for demo recording
        # Initialize core components: environment, keyboard listener, progress bar
        self.init_environment()
        self.init_keyboard_listener()
        self.init_progress_bar()

        # Main recording loop (run until enough successful demos are collected)
        try:
            while self.success_count < self.successes_needed:
                if self.recording_started:
                    self.process_env_step()
        finally:
            # Clean up resources to avoid leaks
            if self.key_listener:
                self.key_listener.stop()
            if self.pbar:
                self.pbar.close()
            if self.env:
                pass  # Close environment if it has a close method

        # Save all collected demo data
        self.save_demos()


def main(_):
    # Main function to initialize and run demo recorder
    recorder = DemoRecorder(exp_name=FLAGS.exp_name, successes_needed=FLAGS.successes_needed)
    recorder.run()


if __name__ == "__main__":
    app.run(main)

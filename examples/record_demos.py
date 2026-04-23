import os
import copy
import pickle as pkl
import datetime
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
    """Class to handle recording of successful demonstration trajectories.
    
    Attributes:
        exp_name: Name of the experiment
        successes_needed: Number of successful demonstrations to collect
        recording_started: Flag indicating if recording is active
        env: Environment instance for demonstration collection
        current_obs: Current observation from the environment
        current_info: Current info dictionary from environment step
        trajectory: List of transitions for current episode
        all_transitions: All transitions from successful demonstrations
        success_count: Count of successfully recorded demonstrations
        pbar: Progress bar for tracking collection progress
        key_listener: Keyboard listener for recording control
    """

    def __init__(self, exp_name: str, successes_needed: int) -> None:
        """Initialize DemoRecorder instance.
        
        Args:
            exp_name: Name of the experiment
            successes_needed: Target number of successful demonstrations
        """
        self.exp_name = exp_name
        self.successes_needed = successes_needed
        self.recording_started = False
        self.env = None
        self.current_obs = None
        self.current_info = None
        self.trajectory: List[Dict[str, Any]] = []
        self.all_transitions: List[Dict[str, Any]] = []
        self.success_count = 0
        self.pbar = None
        self.key_listener = None

    def init_environment(self) -> None:
        """Initialize experiment environment based on experiment name."""
        assert self.exp_name in CONFIG_MAPPING, f"Experiment {self.exp_name} not found in CONFIG_MAPPING"
        config = CONFIG_MAPPING[self.exp_name]()
        self.env = config.get_environment(fake_env=False, classifier=True)
        self.current_obs, self.current_info = self.env.reset()
        print("✅ Environment reset completed")

    def init_keyboard_listener(self) -> None:
        """Initialize keyboard listener to control recording state (toggle with 'b')."""

        def on_press(key: keyboard.Key) -> None:
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

    def init_progress_bar(self) -> None:
        """Initialize progress bar to track successful demo collection progress."""
        self.pbar = tqdm(
            total=self.successes_needed,
            desc=f"Recording: {'ON' if self.recording_started else 'OFF'}",
            ncols=80,
            leave=True,
        )

    def process_env_step(self) -> None:
        """Process single environment step (only active when recording is enabled)."""
        action_shape = self.env.action_space.sample().shape
        actions = np.zeros(action_shape)

        next_obs, rew, done, _, info = self.env.step(actions)

        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy({
            "observations": self.current_obs,
            "actions": actions,
            "next_observations": next_obs,
            "rewards": rew,
            "masks": 1.0 - done,
            "dones": done,
            "infos": info,
        })

        self.trajectory.append(transition)
        self.current_obs = next_obs

        if done:
            self.handle_episode_end(info, done)

    def handle_episode_end(self, info: Dict[str, Any], done: bool) -> None:
        """Handle post-episode logic including saving successful trajectories.
        
        Args:
            info: Info dictionary from environment step
            done: Flag indicating episode termination
        """
        was_recording = self.recording_started

        if self.recording_started:
            self.recording_started = False
            print("\n[Auto-stopped] Episode finished. Press 'b' to start next recording.")
            self.pbar.set_description(f"Recording: OFF")
            self.pbar.refresh()

        if info.get("succeed", False) and was_recording:
            self.all_transitions.extend(copy.deepcopy(self.trajectory))
            self.success_count += 1
            self.pbar.update(1)

        self.trajectory = []
        self.current_obs, self.current_info = self.env.reset()
        print("✅ Environment reset completed. Press 'b' to start next recording.")

    def save_demos(self) -> None:
        """Save collected demonstration transitions to pickle file with timestamp."""
        save_dir = os.path.join(os.path.dirname(__file__), "experiments", self.exp_name, "demo_data")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(save_dir, f"{self.exp_name}_{self.successes_needed}_demos_{timestamp}.pkl")

        with open(file_path, "wb") as f:
            pkl.dump(self.all_transitions, f)

        print(f"\n🎉 Successfully saved {self.successes_needed} demos to:")
        print(f"   {file_path}")

    def run(self) -> None:
        """Main execution flow for demonstration recording."""
        self.init_environment()
        self.init_keyboard_listener()
        self.init_progress_bar()

        try:
            while self.success_count < self.successes_needed:
                if self.recording_started:
                    self.process_env_step()
        finally:
            if self.key_listener:
                self.key_listener.stop()
            if self.pbar:
                self.pbar.close()

        self.save_demos()


def main(_: Any) -> None:
    """Main function to initialize and run DemoRecorder.
    
    Args:
        _: Unused argument from absl app
    """
    recorder = DemoRecorder(exp_name=FLAGS.exp_name, successes_needed=FLAGS.successes_needed)
    recorder.run()


if __name__ == "__main__":
    app.run(main)

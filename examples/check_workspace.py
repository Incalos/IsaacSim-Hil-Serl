import os
import yaml
import requests
import numpy as np
from absl import app, flags
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Experiment name, corresponding to config folder.")
flags.DEFINE_string("yaml_path", "so101_params.yaml", "Output YAML filename.")
flags.DEFINE_integer("save_interval", 10, "Save bounding box every N steps.")


def save_bounding_box(yaml_file, bounding_box):
    """Load existing YAML if present, merge in bounding_box, and write back."""
    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data["bounding_box"] = bounding_box
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment {FLAGS.exp_name} not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, FLAGS.yaml_path)
    # Initialize workspace bounds to extreme values for min/max tracking.
    min_translation = np.array([float("inf")] * 3)
    max_translation = np.array([float("-inf")] * 3)
    min_rotation = np.array([float("inf")] * 3)
    max_rotation = np.array([float("-inf")] * 3)
    env.reset()
    step_count = 0
    env.unwrapped.xyz_bounding_box = None
    env.unwrapped.rpy_bounding_box = None
    try:
        while True:
            actions = np.zeros(env.action_space.sample().shape)
            env.step(actions)
            # Fetch end-effector pose (x, y, z, roll, pitch, yaw) from robot server.
            response = requests.post(env.unwrapped.url + "/get_eef_poses_euler").json()
            pose_data = np.array(response)
            curr_pos = pose_data[:3]
            curr_euler = pose_data[3:]
            # Expand bounding box to include current pose.
            min_translation = np.minimum(min_translation, curr_pos)
            max_translation = np.maximum(max_translation, curr_pos)
            min_rotation = np.minimum(min_rotation, curr_euler)
            max_rotation = np.maximum(max_rotation, curr_euler)
            step_count += 1
            # Periodically persist current bounds to YAML.
            if step_count % FLAGS.save_interval == 0:
                bounding_box = {
                    "min_translation": min_translation.tolist(),
                    "max_translation": max_translation.tolist(),
                    "min_rotation": min_rotation.tolist(),
                    "max_rotation": max_rotation.tolist(),
                }
                save_bounding_box(yaml_path, bounding_box)
                print("Updated bounds:")
                print(f"min_translation: {min_translation}")
                print(f"max_translation: {max_translation}")
                print(f"min_rotation: {min_rotation}")
                print(f"max_rotation: {max_rotation}")
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Recording stopped.")
    finally:
        # Persist final bounding box if any pose was recorded.
        if not np.isinf(min_translation[0]):
            final_bbox = {
                "min_translation": min_translation.tolist(),
                "max_translation": max_translation.tolist(),
                "min_rotation": min_rotation.tolist(),
                "max_rotation": max_rotation.tolist(),
            }
            save_bounding_box(yaml_path, final_bbox)
            print(f"Final workspace bounds saved to: {yaml_path}")
        env.close()


if __name__ == "__main__":
    app.run(main)

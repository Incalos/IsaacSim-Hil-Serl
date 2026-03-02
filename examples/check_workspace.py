import os
import yaml
import requests
import numpy as np
from absl import app, flags
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Experiment name, corresponding to config folder.")
flags.DEFINE_string("yaml_name", "so101_params.yaml", "Output YAML filename.")
flags.DEFINE_integer("save_interval", 10, "Save bounding box every N steps.")


# Merge and save bounding box data to YAML file
def save_bounding_box(yaml_file, bounding_box):
    data = yaml.safe_load(open(yaml_file, "r")) or {} if os.path.exists(yaml_file) else {}
    data["bounding_box"] = bounding_box
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)

    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main(_):
    # Validate experiment name exists in config mapping
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment {FLAGS.exp_name} not found."

    # Initialize experiment configuration and environment
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    # Set up output path for YAML file
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', FLAGS.exp_name)
    yaml_path = os.path.join(base_dir, FLAGS.yaml_name)

    # Initialize bounding box tracking with extreme values
    min_translation = np.array([float("inf")] * 3)
    max_translation = np.array([float("-inf")] * 3)
    min_rotation = np.array([float("inf")] * 3)
    max_rotation = np.array([float("-inf")] * 3)

    # Reset environment and initialize bounding box attributes
    env.reset()
    step_count = 0
    env.unwrapped.xyz_bounding_box = None
    env.unwrapped.rpy_bounding_box = None

    try:
        # Main loop for tracking end-effector pose
        while True:
            # Generate zero action and step environment
            actions = np.zeros(env.action_space.sample().shape)
            env.step(actions)

            # Fetch end-effector pose (x,y,z,roll,pitch,yaw) from robot server
            response = requests.post(env.unwrapped.url + "/get_eef_poses_euler").json()
            pose_data = np.array(response)
            curr_pos = pose_data[:3]
            curr_euler = pose_data[3:]

            # Update bounding box with current pose values
            min_translation = np.minimum(min_translation, curr_pos)
            max_translation = np.maximum(max_translation, curr_pos)
            min_rotation = np.minimum(min_rotation, curr_euler)
            max_rotation = np.maximum(max_rotation, curr_euler)

            step_count += 1

            # Save bounding box at specified interval
            if step_count % FLAGS.save_interval == 0:
                bounding_box = {
                    "min_translation": min_translation.tolist(),
                    "max_translation": max_translation.tolist(),
                    "min_rotation": min_rotation.tolist(),
                    "max_rotation": max_rotation.tolist(),
                }
                save_bounding_box(yaml_path, bounding_box)

                # Print updated bounds to console
                print("Updated bounds:")
                print(f"min_translation: {min_translation}")
                print(f"max_translation: {max_translation}")
                print(f"min_rotation: {min_rotation}")
                print(f"max_rotation: {max_rotation}")

    # Handle keyboard interrupt (Ctrl+C)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Recording stopped.")

    # Save final bounding box and clean up
    finally:
        # Only save if valid pose data was recorded
        if not np.isinf(min_translation[0]):
            final_bbox = {
                "min_translation": min_translation.tolist(),
                "max_translation": max_translation.tolist(),
                "min_rotation": min_rotation.tolist(),
                "max_rotation": max_rotation.tolist(),
            }
            save_bounding_box(yaml_path, final_bbox)
            print(f"Final workspace bounds saved to: {yaml_path}")

        # Close environment connection
        env.close()


if __name__ == "__main__":
    app.run(main)

import os
import yaml
import requests
import numpy as np
from typing import Dict, List, Any
from absl import app, flags
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Experiment name, corresponding to config folder.")
flags.DEFINE_integer("save_interval", 10, "Save bounding box every N steps.")


def save_bounding_box(yaml_file: str, bounding_box: Dict[str, List[float]]) -> None:
    """Merge and save bounding box data to YAML file.
    
    Args:
        yaml_file: Path to YAML file for saving
        bounding_box: Dictionary containing bounding box limits
    """
    data = yaml.safe_load(open(yaml_file, "r")) or {} if os.path.exists(yaml_file) else {}
    data["bounding_box"] = bounding_box
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)

    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main(_: Any) -> None:
    """Main function to track end-effector pose and save workspace bounding box.
    
    Args:
        _: Unused argument from absl app
    """
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment {FLAGS.exp_name} not found."

    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, classifier=False)

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", FLAGS.exp_name)
    yaml_path = os.path.join(base_dir, "exp_params.yaml")

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

            response = requests.post(env.unwrapped.flask_server + "/get_eef_poses_euler").json()
            pose_data = np.array(response)
            curr_pos = pose_data[:3]
            curr_euler = pose_data[3:]

            min_translation = np.minimum(min_translation, curr_pos)
            max_translation = np.maximum(max_translation, curr_pos)
            min_rotation = np.minimum(min_rotation, curr_euler)
            max_rotation = np.maximum(max_rotation, curr_euler)

            step_count += 1

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

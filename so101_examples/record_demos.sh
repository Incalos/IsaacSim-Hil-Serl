cd ..
source .venv/bin/activate
uv run scripts/teleoperation/so101_teleoperation.py --task SO101-PickOranges --teleop_device=keyboard --num_envs=1 --dataset_file=./datasets/dataset.hdf5 --device cuda --enable_cameras --record  --use_lerobot_recorder
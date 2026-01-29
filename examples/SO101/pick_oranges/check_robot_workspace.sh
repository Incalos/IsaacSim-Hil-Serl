cd ../../../
source .venv/bin/activate
uv run scripts/check_robot_workspace/so101_workspace.py --task=SO101-PickOranges --teleop_device=keyboard --num_envs=1 --device=cuda --enable_cameras --quality
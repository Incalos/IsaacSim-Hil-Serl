cd ../../../
source .venv/bin/activate
# Stage 1
uv run examples/train_reward_classifier.py --exp_name=so101_pick_oranges --checkpoint_name=stage1.pth --data_path=stage_1
# Stage 2
# uv run examples/train_reward_classifier.py --exp_name=so101_pick_oranges --checkpoint_name=stage2.pth --data_path=stage_2
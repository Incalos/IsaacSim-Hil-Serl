cd ../../../
source .venv/bin/activate
uv run examples/train_rlpd.py --exp_name=so101_pick_oranges --checkpoint_path=examples/experiments/so101_pick_oranges/checkpoints --ip=localhost --learner 
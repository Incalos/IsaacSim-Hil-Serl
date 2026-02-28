cd ../../../
source .venv/bin/activate
uv run examples/train_rlpd.py --exp_name=so101_reach_orange --checkpoint_path=examples/experiments/so101_reach_orange/checkpoints --ip=localhost --actor --eval_checkpoint_step=49000 --eval_n_trajs=10
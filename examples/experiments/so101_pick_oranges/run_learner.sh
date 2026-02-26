cd ../../../
source .venv/bin/activate
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x16 &
uv run examples/train_rlpd.py --exp_name=so101_pick_oranges --checkpoint_path=examples/experiments/so101_pick_oranges/checkpoints --ip=localhost --learner 
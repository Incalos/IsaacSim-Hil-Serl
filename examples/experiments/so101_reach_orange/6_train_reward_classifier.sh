cd ../../../
source .venv/bin/activate
uv run examples/train_reward_classifier.py --exp_name=so101_reach_orange --checkpoint_name=checkpoint.pth --data_path=classifier_data
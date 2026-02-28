cd ../../../
source .venv/bin/activate
uv run examples/record_success_fail.py --exp_name=so101_reach_orange --successes_needed=100  --save_interval=5 --save_path=classifier_data
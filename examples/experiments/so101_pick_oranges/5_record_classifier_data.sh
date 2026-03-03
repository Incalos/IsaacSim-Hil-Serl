cd ../../../
source .venv/bin/activate
# Stage 1
# uv run examples/record_success_fail.py --exp_name=so101_pick_oranges --successes_needed=100  --save_interval=5 --save_path=stage_1
# Stage 2
uv run examples/record_success_fail.py --exp_name=so101_pick_oranges --successes_needed=100  --save_interval=5 --save_path=stage_2
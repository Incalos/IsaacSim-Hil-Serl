cd ..
source .venv/bin/activate
uv run robot_envs/isaacsim_run/so101_env.py --task SO101-PickOranges --device cpu --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"
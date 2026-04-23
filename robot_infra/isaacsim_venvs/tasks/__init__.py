import gymnasium


gymnasium.register(
    id="SO101_GRASP_ORANGE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.so101_grasp_orange_env_cfg:GraspOrangeEnvCfg"},
)
import gymnasium as gym
from .droid_environment import EnvCfg as DroidEnvCfg
from .yam_environment import YamEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

gym.register(
    id="DROID",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": DroidEnvCfg,
    },
    disable_env_checker=True,
)

# Second embodiment sharing the same scenes and the same ManagerBasedRLEnv; only the robot, its
# action/observation terms and the perception camera differ. See yam_environment.py.
gym.register(
    id="YAM",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": YamEnvCfg,
    },
    disable_env_checker=True,
)

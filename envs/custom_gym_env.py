# envs/custom_gym_envs.py
from .custom_channel_env import PyMARLCustomEnv

def env_fn(env_args):
    return PyMARLCustomEnv(env_args)

ENV_REGISTRY = {}
ENV_REGISTRY["custom_6g_env"] = env_fn

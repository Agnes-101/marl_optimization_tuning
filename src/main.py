# src/main.py
import sys
import os

# Assuming your project root is the current working directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print("Project root:", project_root)
print("sys.path:", sys.path)

import argparse
import yaml
from envs.custom_gym_env import ENV_REGISTRY
from envs.custom_channel_env import PyMARLCustomEnv

def main(config, env_config):
    # Load configuration from YAML.
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    print("Configuration:")
    print(config_data)
    
    # Instantiate the environment.
    env_fn = ENV_REGISTRY.get(env_config)
    if env_fn is None:
        raise ValueError(f"Environment {env_config} is not registered.")
    env_args = config_data.get("env_args", {})
    env = env_fn(env_args)
    
    # For demonstration, run one episode.
    state = env.reset()
    print("Initial State:")
    print(state)
    actions = [env.action_space.sample() for _ in range(env.num_users)]
    step_data = env.step(actions)
    print("Step Data:")
    print(step_data)
    print("Main execution complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/qmix_custom.yaml")
    parser.add_argument("--env-config", type=str, default="custom_6g_env")
    args = parser.parse_args()
    main(args.config, args.env_config)

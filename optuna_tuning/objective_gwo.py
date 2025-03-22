# File: optuna_tuning/objective_gwo.py
import sys
import os
import optuna
import numpy as np
import time

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.gwo import GWOOptimization

def train_gwo(trial, fixed_env_args):
    # Hyperparameters to tune
    params = {
        'population_size': trial.suggest_int('population_size', 30, 100),  # Match PFO range
        'iterations': trial.suggest_int('iterations', 50, 300),
        'a_initial': trial.suggest_float('a_initial', 1.5, 2.5),
        'a_decay': trial.suggest_float('a_decay', 0.9, 0.99),
        'alpha': trial.suggest_float('alpha', 0.01, 0.1),  # Added to match PFO
        'beta': trial.suggest_float('beta', 0.01, 0.1),    # Added to match PFO
    }

    # Environment setup with seeding
    env = PyMARLCustomEnv(fixed_env_args)
    env.seed(42)  # Requires seed() method in your environment class
    env.reset()

    # Initialize GWO
    gwo = GWOOptimization(
        num_users=env.num_users,
        num_cells=env.action_space.n,
        env=env.env,  # Keep if your env has wrapper structure
        swarm_size=params['population_size'],
        iterations=params['iterations'],
        a_initial=params['a_initial'],
        a_decay=params['a_decay'],
        seed=42  # Add if your GWO implementation supports seeding
    )
    
    # Optimization
    start_time = time.time()
    best_solution = gwo.optimize()
    
    # Evaluation
    metrics = evaluate_detailed_solution(
        env.env, 
        best_solution,
        alpha=params['alpha'],
        beta=params['beta']
    )
    
    # Log metrics
    trial.set_user_attr("execution_time", time.time() - start_time)
    for metric, value in metrics.items():
        trial.set_user_attr(metric, value)

    return metrics["fitness"]

def objective(trial):
    # Fixed environment configuration (identical to PFO)
    env_args = {
        "num_users": 60,
        "episode_limit": 50,
        "macro_positions": [[50, 50]],
        "small_positions": [[20, 20], [20, 80], [80, 20], [80, 80]],
        "cell_capacity": 15
    }
    return train_gwo(trial, env_args)

if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///optuna.db",  # Same as PFO
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100, n_jobs=-1)
    
    print("\n=== Best Parameters ===")
    for key, value in study.best_params.items():
        print(f"{key}: {value:.4f}")
    
    best_trial = study.best_trial
    print("\n=== Best Metrics ===")
    print(f"Fitness: {best_trial.value:.4f}")
    print(f"SINR: {best_trial.user_attrs['average_sinr']:.2f} dB")
    print(f"Throughput: {best_trial.user_attrs['average_throughput']:.2f} Mbps")
    print(f"Time: {best_trial.user_attrs['execution_time']:.1f}s")

# import sys
# import os

# # Add the project root to Python's path manually
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root)

# import optuna
# import json
# import yaml
# from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
# from algorithms.gwo import GWOOptimization

# def train_gwo(iterations, swarm_size, a_initial, a_decay, num_episodes=10, seed=42):
#     # Load environment config
#     with open("config/env_config.yaml", 'r') as f:
#         env_args = yaml.safe_load(f)
    
#     # Initialize environment with seed
#     env = PyMARLCustomEnv(env_args, seed=seed)
#     num_users = env.num_users
#     num_cells = env.action_space.n
    
#     # Initialize GWO with tuned parameters
#     gwo = GWOOptimization(
#         num_users, num_cells, env.env,
#         swarm_size=swarm_size,
#         iterations=iterations,
#         a_initial=a_initial,
#         a_decay=a_decay
#     )
    
#     best_solution = gwo.optimize()
#     metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
#     return metrics["fitness"]

# def objective(trial):
#     # Hyperparameters to tune
#     iterations = trial.suggest_int('gwo_iterations', 30, 70)
#     swarm_size = trial.suggest_int('gwo_swarm_size', 20, 40)
#     a_initial = trial.suggest_float('gwo_a_initial', 2.0, 2.5)
#     a_decay = trial.suggest_float('gwo_a_decay', 0.95, 0.99)
    
#     final_reward = train_gwo(iterations, swarm_size, a_initial, a_decay, num_episodes=10)
#     print(f"GWO Trial {trial.number}: Reward={final_reward:.3f}")
#     return final_reward

# if __name__ == '__main__':
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=50, n_jobs=-1)  # Increased trials + parallelism
    
#     # Save best parameters
#     with open("optuna_tuning/best_params_gwo.json", 'w') as f:
#         json.dump(study.best_params, f)
    
#     print("Best GWO parameters:", study.best_params)
#     print("Best GWO reward:", study.best_value)


# # File: optuna_tuning/objective_pfo.py
import sys
import os
import optuna
import numpy as np
import time

#
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.pfo import PolarFoxOptimization

def train_pfo(trial, fixed_env_args):
    # Hyperparameters to tune
    seed = 42  # Fixed seed for reproducibility
    np.random.seed(seed)
    # torch.manual_seed(seed)  # If using PyTorch
    # random.seed(seed)

    params = {
        'population_size': trial.suggest_int('population_size', 30, 100),
        'iterations': trial.suggest_int('iterations', 50, 300),
        'mutation_factor': trial.suggest_float('mutation_factor', 0.1, 0.5),
        'initial_jump_power': trial.suggest_float('initial_jump_power', 0.5, 2.0),
        'alpha': trial.suggest_float('alpha', 0.01, 0.1),# before 0.05-0.3
        'beta': trial.suggest_float('beta', 0.01, 0.1),
        'group1_weight': trial.suggest_float('group1_weight', 0.3, 0.6),  # Explorer group before(0.1-0.4)
        'group2_weight': trial.suggest_float('group2_weight', 0.25, 0.4),
        'group3_weight': trial.suggest_float('group3_weight', 0.25, 0.4),  
        'group4_weight': trial.suggest_float('group4_weight', 0.1, 0.4),# Conservative group
    }

    # Environment setup (fixed for consistency)
    env = PyMARLCustomEnv(fixed_env_args)
    env.seed(42)  
    env.reset()
    
    # Initialize PFO with tuned parameters
    pfo = PolarFoxOptimization(
        num_users=env.num_users,
        num_cells=env.action_space.n,
        env=env,
        population_size=params['population_size'],
        iterations=params['iterations'],
        mutation_factor=params['mutation_factor'],
        seed=42  # Fixed for reproducibility
    )
    
    # Override group weights dynamically
    pfo.group_weights = [
        params['group1_weight'],      # Group 1 (Explorer)
        # 0.25,
        # 0.25,
        params['group2_weight'],# 0.25,                         # Group 2 (Balanced)
        params['group4_weight'], # 0.25,                         # Group 3 (Follower)
        params['group4_weight']       # Group 4 (Conservative)
    ]
    
    # Optimization
    start_time = time.time()
    best_solution = pfo.optimize()
    exec_time = time.time() - start_time
    
    # Run optimization
    best_solution = pfo.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, 
                                       alpha=params['alpha'], 
                                       beta=params['beta'])
    
    # Logging
    trial.set_user_attr("execution_time", exec_time)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    return metrics["fitness"]

def objective(trial):
    # Fixed environment configuration
    env_args = {
        "num_users": 60,
        "episode_limit": 50,
        "macro_positions": [[50, 50]],
        "small_positions": [[20, 20], [20, 80], [80, 20], [80, 80]],
        "cell_capacity": 15,
        "seed": 42
    }
    
    return train_pfo(trial, env_args)

if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///optuna.db",
        sampler=optuna.samplers.TPESampler(n_startup_trials=20, seed=42),
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
    




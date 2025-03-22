# File: optuna_tuning/objective_hs.py
import sys
import os
import optuna
import time

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.hs import HarmonySearchOptimization

def train_hs(trial, fixed_env_args):
    # Hyperparameters to tune
    params = {
        'population_size': trial.suggest_int('population_size', 30, 100),  # Match other algorithms
        'iterations': trial.suggest_int('iterations', 50, 300),            # Match other algorithms
        'HMCR': trial.suggest_float('HMCR', 0.7, 0.95),
        'PAR': trial.suggest_float('PAR', 0.2, 0.5),
        'alpha': trial.suggest_float('alpha', 0.01, 0.1),  # Reward parameter
        'beta': trial.suggest_float('beta', 0.01, 0.1),    # Reward parameter
    }

    # Environment setup with seeding
    env = PyMARLCustomEnv(fixed_env_args)
    env.seed(42)  # Requires seed() method in environment
    env.reset()

    # Initialize Harmony Search
    hs = HarmonySearchOptimization(
        num_users=env.num_users,
        num_cells=env.action_space.n,
        env=env.env,
        memory_size=params['population_size'],
        iterations=params['iterations'],
        HMCR=params['HMCR'],
        PAR=params['PAR'],
        seed=42  # Add if your HS implementation supports seeding
    )
    
    # Optimization
    start_time = time.time()
    best_solution = hs.optimize()
    
    # Evaluation (using tuned alpha/beta from reward params)
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
    # Fixed environment configuration (identical to others)
    env_args = {
        "num_users": 60,
        "episode_limit": 50,
        "macro_positions": [[50, 50]],
        "small_positions": [[20, 20], [20, 80], [80, 20], [80, 80]],
        "cell_capacity": 15
    }
    return train_hs(trial, env_args)

if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///optuna.db",  # Same storage
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
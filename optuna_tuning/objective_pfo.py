# File: optuna_tuning/objective_pfo.py
import sys
import os

# Add project root to sys.path so that 'envs' can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.pfo import PolarFoxOptimization
import numpy as np

def train_pfo(lr, pfo_jump_rate, alpha, beta, num_episodes=10):
    """
    Sets up the environment and runs the PFO optimization algorithm.
    
    Parameters:
      lr: Learning rate (even if not used by PFO directly, can be tuned for consistency).
      pfo_jump_rate: The jump rate used by the PFO algorithm.
      alpha: Weight for the normalized distance penalty.
      beta: Weight for the load imbalance penalty.
      num_episodes: Number of episodes (or iterations) for a short training run.
      
    Returns:
      final_reward: The overall fitness (reward) computed by the detailed evaluation.
    """
    env_args = {
        "num_users": 60,
        "episode_limit": 50,
        "macro_positions": [[50, 50]],
        "small_positions": [[20, 20], [20, 80], [80, 20], [80, 80]],
        "cell_capacity": 15
    }
    # Create the Gym-compatible environment.
    env = PyMARLCustomEnv(env_args)
    num_users = env.num_users
    num_cells = env.action_space.n

    # Instantiate the PFO algorithm with given jump rate (other parameters can be fixed or tuned separately).
    pfo = PolarFoxOptimization(num_users, num_cells, env, population_size=30, iterations=10, jump_rate=pfo_jump_rate, follow_rate=0.3)
    best_solution = pfo.optimize()

    # Use the detailed evaluation function that now accepts alpha and beta.
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=alpha, beta=beta)
    final_reward = metrics["fitness"]
    return final_reward

def objective(trial):
    """
    Optuna objective function for tuning PFO hyperparameters along with alpha and beta.
    """
    # Suggest hyperparameters for PFO and the penalty weights.
    lr = trial.suggest_loguniform('lr_pfo', 1e-5, 1e-2)
    pfo_jump_rate = trial.suggest_float('pfo_jump_rate', 0.1, 0.5)
    alpha = trial.suggest_float('alpha', 0.05, 0.2)
    beta = trial.suggest_float('beta', 0.05, 0.2)
    
    final_reward = train_pfo(lr, pfo_jump_rate, alpha, beta, num_episodes=10)
    print(f"Trial {trial.number}: lr={lr:.5f}, pfo_jump_rate={pfo_jump_rate:.3f}, alpha={alpha:.3f}, beta={beta:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best parameters:", study.best_params)
    print("Best reward:", study.best_value)

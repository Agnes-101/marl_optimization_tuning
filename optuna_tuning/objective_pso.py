# optuna_tuning/objective_pso.py
import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.pso import PSOOptimization

def train_pso(lr, c1, c2, w, num_episodes=10):
    env_args = {
        "num_users": 60,
        "episode_limit": 50,
        "macro_positions": [[50, 50]],
        "small_positions": [[20, 20], [20, 80], [80, 20], [80, 80]],
        "cell_capacity": 15
    }
    env = PyMARLCustomEnv(env_args)
    num_users = env.num_users
    num_cells = env.action_space.n

    # Instantiate the PSO optimizer.
    pso = PSOOptimization(num_users, num_cells, env.env, swarm_size=30, iterations=10, c1=c1, c2=c2, w=w)
    best_solution = pso.optimize()

    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    lr = trial.suggest_loguniform('lr_pso', 1e-5, 1e-2)
    c1 = trial.suggest_float('c1', 0.5, 2.0)
    c2 = trial.suggest_float('c2', 0.5, 2.0)
    w = trial.suggest_float('w', 0.1, 1.0)
    final_reward = train_pso(lr, c1, c2, w, num_episodes=10)
    print(f"PSO Trial {trial.number}: lr={lr:.5f}, c1={c1:.3f}, c2={c2:.3f}, w={w:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best PSO parameters:", study.best_params)
    print("Best PSO reward:", study.best_value)

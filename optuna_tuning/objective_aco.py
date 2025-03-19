# optuna_tuning/objective_aco.py
import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.aco import ACOOptimization

def train_aco(alpha, beta, evaporation_rate, num_episodes=10):
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

    aco = ACOOptimization(num_users, num_cells, env.env, ants=30, iterations=10, evaporation_rate=evaporation_rate, alpha=alpha, beta=beta)
    best_solution = aco.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    alpha_param = trial.suggest_float('aco_alpha', 0.5, 2.0)
    beta_param = trial.suggest_float('aco_beta', 0.5, 2.0)
    evaporation_rate = trial.suggest_float('evaporation_rate', 0.05, 0.2)
    final_reward = train_aco(alpha_param, beta_param, evaporation_rate, num_episodes=10)
    print(f"ACO Trial {trial.number}: aco_alpha={alpha_param:.3f}, aco_beta={beta_param:.3f}, evaporation_rate={evaporation_rate:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best ACO parameters:", study.best_params)
    print("Best ACO reward:", study.best_value)

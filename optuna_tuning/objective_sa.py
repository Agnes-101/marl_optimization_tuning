# optuna_tuning/objective_sa.py
import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.sa import SAOptimization

def train_sa(initial_temp, cooling_rate, num_episodes=10):
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

    sa = SAOptimization(num_users, num_cells, env.env, iterations=50, initial_temp=initial_temp, cooling_rate=cooling_rate)
    best_solution = sa.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    initial_temp = trial.suggest_float('initial_temp', 50, 150)
    cooling_rate = trial.suggest_float('cooling_rate', 0.90, 0.99)
    final_reward = train_sa(initial_temp, cooling_rate, num_episodes=10)
    print(f"SA Trial {trial.number}: initial_temp={initial_temp:.2f}, cooling_rate={cooling_rate:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best SA parameters:", study.best_params)
    print("Best SA reward:", study.best_value)

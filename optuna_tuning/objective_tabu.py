import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.tabu import TabuSearchOptimization

def train_tabu(iterations, tabu_size, num_episodes=10):
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
    tabu = TabuSearchOptimization(num_users, num_cells, env.env, iterations=iterations, tabu_size=tabu_size)
    best_solution = tabu.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    iterations = trial.suggest_int('iterations_tabu', 30, 70)
    tabu_size = trial.suggest_int('tabu_size', 5, 15)
    final_reward = train_tabu(iterations, tabu_size, num_episodes=10)
    print(f"Tabu Trial {trial.number}: iterations={iterations}, tabu_size={tabu_size} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best Tabu parameters:", study.best_params)
    print("Best Tabu reward:", study.best_value)

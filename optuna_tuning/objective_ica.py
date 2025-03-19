import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.ica import ICAOptimization

def train_ica(population_size, imperialist_count, iterations, num_episodes=10):
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
    ica = ICAOptimization(num_users, num_cells, env.env, population_size=population_size, imperialist_count=imperialist_count, iterations=iterations)
    best_solution = ica.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    population_size = trial.suggest_int('population_size_ica', 20, 40)
    imperialist_count = trial.suggest_int('imperialist_count', 3, 10)
    iterations = trial.suggest_int('iterations_ica', 30, 70)
    final_reward = train_ica(population_size, imperialist_count, iterations, num_episodes=10)
    print(f"ICA Trial {trial.number}: population_size={population_size}, imperialist_count={imperialist_count}, iterations={iterations} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best ICA parameters:", study.best_params)
    print("Best ICA reward:", study.best_value)

import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.de import DEOptimization

def train_de(population_size, iterations, F, CR, num_episodes=10):
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
    de = DEOptimization(num_users, num_cells, env.env, population_size=population_size, iterations=iterations, F=F, CR=CR)
    best_solution = de.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    population_size = trial.suggest_int('population_size_de', 20, 40)
    iterations = trial.suggest_int('iterations_de', 30, 70)
    F = trial.suggest_float('F_de', 0.5, 1.0)
    CR = trial.suggest_float('CR_de', 0.5, 1.0)
    final_reward = train_de(population_size, iterations, F, CR, num_episodes=10)
    print(f"DE Trial {trial.number}: population_size={population_size}, iterations={iterations}, F={F:.3f}, CR={CR:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best DE parameters:", study.best_params)
    print("Best DE reward:", study.best_value)

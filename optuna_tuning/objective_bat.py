import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.bat import BatOptimization

def train_bat(population_size, iterations, freq_min, freq_max, alpha_bat, gamma_bat, num_episodes=10):
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
    bat = BatOptimization(num_users, num_cells, env.env, population_size=population_size, iterations=iterations,
                          freq_min=freq_min, freq_max=freq_max, alpha=alpha_bat, gamma=gamma_bat)
    best_solution = bat.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    population_size = trial.suggest_int('population_size_bat', 20, 40)
    iterations = trial.suggest_int('iterations_bat', 30, 70)
    freq_min = trial.suggest_float('freq_min', 0, 1)
    freq_max = trial.suggest_float('freq_max', 1, 3)
    alpha_bat = trial.suggest_float('alpha_bat', 0.7, 1.0)
    gamma_bat = trial.suggest_float('gamma_bat', 0.7, 1.0)
    final_reward = train_bat(population_size, iterations, freq_min, freq_max, alpha_bat, gamma_bat, num_episodes=10)
    print(f"Bat Trial {trial.number}: pop_size={population_size}, iterations={iterations}, freq_min={freq_min:.2f}, freq_max={freq_max:.2f}, alpha_bat={alpha_bat:.3f}, gamma_bat={gamma_bat:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best Bat parameters:", study.best_params)
    print("Best Bat reward:", study.best_value)

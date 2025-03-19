import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.fa import FireflyOptimization

def train_fa(population_size, iterations, beta0, gamma_fa, num_episodes=10):
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
    fa = FireflyOptimization(num_users, num_cells, env.env, population_size=population_size, iterations=iterations, beta0=beta0, gamma=gamma_fa)
    best_solution = fa.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    population_size = trial.suggest_int('population_size_fa', 20, 40)
    iterations = trial.suggest_int('iterations_fa', 30, 70)
    beta0 = trial.suggest_float('beta0', 0.5, 1.5)
    gamma_fa = trial.suggest_float('gamma_fa', 0.5, 1.5)
    final_reward = train_fa(population_size, iterations, beta0, gamma_fa, num_episodes=10)
    print(f"FA Trial {trial.number}: pop_size={population_size}, iterations={iterations}, beta0={beta0:.3f}, gamma_fa={gamma_fa:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best FA parameters:", study.best_params)
    print("Best FA reward:", study.best_value)

# optuna_tuning/objective_ga.py
import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.ga import GAOptimization

def train_ga(lr, mutation_rate, num_episodes=10):
    # Set up environment arguments.
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
    
    # Instantiate the GA optimizer.
    ga = GAOptimization(num_users, num_cells, env.env, population_size=30, generations=10, mutation_rate=mutation_rate)
    best_solution = ga.optimize()
    
    # Evaluate the best solution using the detailed evaluation function.
    # Here, alpha and beta are set to fixed values; you can also tune them.
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    # Suggest hyperparameters for GA.
    lr = trial.suggest_loguniform('lr_ga', 1e-5, 1e-2)  # In GA this might be less relevant
    mutation_rate = trial.suggest_float('mutation_rate', 0.05, 0.2)
    final_reward = train_ga(lr, mutation_rate, num_episodes=10)
    print(f"GA Trial {trial.number}: lr={lr:.5f}, mutation_rate={mutation_rate:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best GA parameters:", study.best_params)
    print("Best GA reward:", study.best_value)

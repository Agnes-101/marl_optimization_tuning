import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.cs import CSOptimization

def train_cs(colony_size, iterations, pa, num_episodes=10):
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
    cs = CSOptimization(num_users, num_cells, env.env, colony_size=colony_size, iterations=iterations, pa=pa)
    best_solution = cs.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    colony_size = trial.suggest_int('colony_size_cs', 20, 40)
    iterations = trial.suggest_int('iterations_cs', 30, 70)
    pa = trial.suggest_float('pa_cs', 0.1, 0.3)
    final_reward = train_cs(colony_size, iterations, pa, num_episodes=10)
    print(f"CS Trial {trial.number}: colony_size={colony_size}, iterations={iterations}, pa={pa:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best CS parameters:", study.best_params)
    print("Best CS reward:", study.best_value)

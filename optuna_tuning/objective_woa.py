import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.woa import WOAOptimization

def train_woa(iterations, swarm_size, num_episodes=10):
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
    woa = WOAOptimization(num_users, num_cells, env.env, swarm_size=swarm_size, iterations=iterations)
    best_solution = woa.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    iterations = trial.suggest_int('iterations_woa', 30, 70)
    swarm_size = trial.suggest_int('swarm_size_woa', 20, 40)
    final_reward = train_woa(iterations, swarm_size, num_episodes=10)
    print(f"WOA Trial {trial.number}: iterations={iterations}, swarm_size={swarm_size} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best WOA parameters:", study.best_params)
    print("Best WOA reward:", study.best_value)

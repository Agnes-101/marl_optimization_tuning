import optuna
from envs.custom_channel_env import PyMARLCustomEnv, evaluate_detailed_solution
from algorithms.hs import HarmonySearchOptimization

def train_hs(memory_size, iterations, HMCR, PAR, num_episodes=10):
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
    hs = HarmonySearchOptimization(num_users, num_cells, env.env, memory_size=memory_size, iterations=iterations, HMCR=HMCR, PAR=PAR)
    best_solution = hs.optimize()
    metrics = evaluate_detailed_solution(env.env, best_solution, alpha=0.1, beta=0.1)
    return metrics["fitness"]

def objective(trial):
    memory_size = trial.suggest_int('memory_size_hs', 20, 40)
    iterations = trial.suggest_int('iterations_hs', 30, 70)
    HMCR = trial.suggest_float('HMCR', 0.7, 0.95)
    PAR = trial.suggest_float('PAR', 0.2, 0.5)
    final_reward = train_hs(memory_size, iterations, HMCR, PAR, num_episodes=10)
    print(f"HS Trial {trial.number}: memory_size={memory_size}, iterations={iterations}, HMCR={HMCR:.3f}, PAR={PAR:.3f} -> Reward: {final_reward:.3f}")
    return final_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best HS parameters:", study.best_params)
    print("Best HS reward:", study.best_value)

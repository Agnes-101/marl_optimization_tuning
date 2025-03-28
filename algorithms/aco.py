# # algorithms/aco.py

# algorithms/aco.py
import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class ACOOptimization:
    def __init__(self, num_users, num_cells, env, ants=30, iterations=50, 
                 evaporation_rate=0.1, alpha=1, beta=2, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.ants = ants
        self.iterations = iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize pheromones with safe values
        self.pheromones = np.ones((num_users, num_cells)) * 0.1
        self.pheromones += self.rng.uniform(0, 0.01, (num_users, num_cells))
        self.best_solution = None
        self.best_fitness = -np.inf

    def fitness(self, solution):
        return max(evaluate_detailed_solution(self.env, solution)["fitness"], 0)

    def construct_solution(self):
        solution = np.zeros(self.num_users, dtype=int)
        for user in range(self.num_users):
            # Numerical stability using log-exp trick
            with np.errstate(divide='ignore'):
                log_pher = np.log(self.pheromones[user] + 1e-20)
            
            log_prob = self.alpha * log_pher
            log_prob -= np.max(log_prob)  # Prevent overflow
            probabilities = np.exp(log_prob) + 1e-10
            probabilities /= probabilities.sum()
            
            solution[user] = self.rng.choice(self.num_cells, p=probabilities)
        return solution

    def update_pheromones(self, solutions):
        # Apply evaporation and ensure positivity
        self.pheromones = np.clip(self.pheromones * (1 - self.evaporation_rate), 1e-10, None)
        
        # Only allow positive pheromone additions
        for sol in solutions:
            fit = self.fitness(sol)
            for user in range(self.num_users):
                self.pheromones[user, sol[user]] += fit
                
        # Post-update clipping
        self.pheromones = np.clip(self.pheromones, 1e-10, 1e5)

    def optimize(self):
        for iteration in range(self.iterations):
            solutions = [self.construct_solution() for _ in range(self.ants)]
            current_best = max(solutions, key=lambda x: self.fitness(x))
            
            if self.fitness(current_best) > self.best_fitness:
                self.best_fitness = self.fitness(current_best)
                self.best_solution = current_best.copy()
            
            # Update with both current and historical best
            self.update_pheromones(solutions + [self.best_solution])
            
            print(f"Iter {iteration+1}: Best Fitness = {self.best_fitness:.4f}")

        return self.best_solution


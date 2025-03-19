# algorithms/aco.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_solution

class ACOOptimization:
    def __init__(self, num_users, num_cells, env, ants=30, iterations=50, evaporation_rate=0.1, alpha=1, beta=2):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.ants = ants
        self.iterations = iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromones = np.ones((num_users, num_cells))
    
    def fitness(self, solution):
        return evaluate_solution(self.env, solution)["fitness"]
    
    def construct_solution(self):
        solution = np.zeros(self.num_users, dtype=int)
        for user in range(self.num_users):
            prob = self.pheromones[user] ** self.alpha
            prob = prob / np.sum(prob)
            solution[user] = np.random.choice(self.num_cells, p=prob)
        return solution
    
    def update_pheromones(self, solutions):
        self.pheromones *= (1 - self.evaporation_rate)
        for sol in solutions:
            fit = self.fitness(sol)
            for user in range(self.num_users):
                self.pheromones[user, sol[user]] += fit
    
    def optimize(self):
        best_solution = None
        best_fit = -np.inf
        for _ in range(self.iterations):
            solutions = [self.construct_solution() for _ in range(self.ants)]
            for sol in solutions:
                fit = self.fitness(sol)
                if fit > best_fit:
                    best_fit = fit
                    best_solution = sol
            self.update_pheromones(solutions)
        return best_solution

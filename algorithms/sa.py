# algorithms/sa.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_solution

class SAOptimization:
    def __init__(self, num_users, num_cells, env, iterations=100, initial_temp=100, cooling_rate=0.95):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.iterations = iterations
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.current_solution = np.random.randint(0, num_cells, size=num_users)
    
    def fitness(self, solution):
        return evaluate_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        current_fit = self.fitness(self.current_solution)
        best_solution = self.current_solution.copy()
        best_fit = current_fit
        for _ in range(self.iterations):
            neighbor = self.current_solution.copy()
            idx = np.random.randint(0, self.num_users)
            neighbor[idx] = np.random.randint(0, self.num_cells)
            neighbor_fit = self.fitness(neighbor)
            delta = neighbor_fit - current_fit
            if delta > 0 or np.random.rand() < np.exp(delta / self.temperature):
                self.current_solution = neighbor
                current_fit = neighbor_fit
                if neighbor_fit > best_fit:
                    best_solution = neighbor.copy()
                    best_fit = neighbor_fit
            self.temperature *= self.cooling_rate
        return best_solution

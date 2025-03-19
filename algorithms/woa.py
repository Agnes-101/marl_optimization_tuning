# algorithms/woa.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_solution

class WOAOptimization:
    def __init__(self, num_users, num_cells, env, swarm_size=30, iterations=50):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.population = [np.random.randint(0, num_cells, size=num_users) for _ in range(swarm_size)]
        self.best_solution = max(self.population, key=self.fitness)
    
    def fitness(self, solution):
        return evaluate_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for t in range(self.iterations):
            a = 2 - t * (2 / self.iterations)
            for i in range(self.swarm_size):
                p = np.random.rand()
                new_solution = self.population[i].copy()
                if p < 0.5:
                    if abs(2 * a * np.random.rand() - a) >= 1:
                        rand_sol = self.population[np.random.randint(self.swarm_size)]
                        for j in range(self.num_users):
                            if np.random.rand() < abs(2 * a * np.random.rand() - a):
                                new_solution[j] = rand_sol[j]
                    else:
                        for j in range(self.num_users):
                            if np.random.rand() < 0.5:
                                new_solution[j] = self.best_solution[j]
                else:
                    for j in range(self.num_users):
                        if np.random.rand() < 0.5:
                            new_solution[j] = self.best_solution[j]
                if self.fitness(new_solution) > self.fitness(self.population[i]):
                    self.population[i] = new_solution
                    if self.fitness(new_solution) > self.fitness(self.best_solution):
                        self.best_solution = new_solution.copy()
        return self.best_solution

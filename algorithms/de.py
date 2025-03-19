# algorithms/de.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_solution

class DEOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, iterations=50, F=0.8, CR=0.9):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.iterations = iterations
        self.F = F
        self.CR = CR
        self.population = [np.random.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
    def fitness(self, solution):
        return evaluate_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for _ in range(self.iterations):
            new_population = []
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = random.sample(indices, 3)
                donor = self.population[a] + self.F * (self.population[b] - self.population[c])
                donor = np.clip(np.round(donor).astype(int), 0, self.num_cells - 1)
                trial = self.population[i].copy()
                for j in range(self.num_users):
                    if np.random.rand() < self.CR:
                        trial[j] = donor[j]
                if self.fitness(trial) > self.fitness(self.population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(self.population[i])
            self.population = new_population
        return max(self.population, key=self.fitness)

# algorithms/pso.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_solution

class PSOOptimization:
    def __init__(self, num_users, num_cells, env, swarm_size=30, iterations=50, c1=1, c2=1, w=0.5):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.positions = [np.random.randint(0, num_cells, size=num_users) for _ in range(swarm_size)]
        self.pbest = self.positions.copy()
        self.gbest = max(self.positions, key=self.fitness)
    
    def fitness(self, solution):
        return evaluate_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.swarm_size):
                new_solution = self.positions[i].copy()
                for j in range(self.num_users):
                    if np.random.rand() < self.c1 * 0.5:
                        new_solution[j] = self.pbest[i][j]
                    if np.random.rand() < self.c2 * 0.5:
                        new_solution[j] = self.gbest[j]
                    if np.random.rand() < self.w * 0.1:
                        new_solution[j] = np.random.randint(0, self.num_cells)
                if self.fitness(new_solution) > self.fitness(self.positions[i]):
                    self.positions[i] = new_solution
                    self.pbest[i] = new_solution
                    if self.fitness(new_solution) > self.fitness(self.gbest):
                        self.gbest = new_solution
        return self.gbest

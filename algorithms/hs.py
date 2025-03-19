# algorithms/hs.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_solution

class HarmonySearchOptimization:
    def __init__(self, num_users, num_cells, env, memory_size=30, iterations=50, HMCR=0.9, PAR=0.3):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.memory_size = memory_size
        self.iterations = iterations
        self.HMCR = HMCR
        self.PAR = PAR
        self.harmony_memory = [np.random.randint(0, num_cells, size=num_users) for _ in range(memory_size)]
    
    def fitness(self, solution):
        return evaluate_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for _ in range(self.iterations):
            new_harmony = np.zeros(self.num_users, dtype=int)
            for j in range(self.num_users):
                if np.random.rand() < self.HMCR:
                    new_harmony[j] = random.choice([h[j] for h in self.harmony_memory])
                    if np.random.rand() < self.PAR:
                        new_harmony[j] = np.random.randint(0, self.num_cells)
                else:
                    new_harmony[j] = np.random.randint(0, self.num_cells)
            worst_index = np.argmin([self.fitness(h) for h in self.harmony_memory])
            if self.fitness(new_harmony) > self.fitness(self.harmony_memory[worst_index]):
                self.harmony_memory[worst_index] = new_harmony
        return max(self.harmony_memory, key=self.fitness)

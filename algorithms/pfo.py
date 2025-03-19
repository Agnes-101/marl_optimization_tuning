# algorithms/pfo.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_detailed_solution

class PolarFoxOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, iterations=50, jump_rate=0.2, follow_rate=0.3):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.iterations = iterations
        self.jump_rate = jump_rate
        self.follow_rate = follow_rate
        self.population = [np.random.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def jump_experience(self, fox):
        new_fox = fox.copy()
        num_jumps = int(len(fox) * self.jump_rate)
        indices = np.random.choice(len(fox), size=num_jumps, replace=False)
        for idx in indices:
            new_fox[idx] = np.random.randint(0, self.num_cells)
        return new_fox
    
    def follow_leader(self, current_fox, best_fox):
        new_fox = current_fox.copy()
        num_follow = int(len(current_fox) * self.follow_rate)
        indices = np.random.choice(len(current_fox), size=num_follow, replace=False)
        for idx in indices:
            new_fox[idx] = best_fox[idx]
        return new_fox
    
    def optimize(self):
        for _ in range(self.iterations):
            fitness_values = [self.fitness(fox) for fox in self.population]
            best_idx = np.argmax(fitness_values)
            best_fox = self.population[best_idx].copy()
            best_fitness = fitness_values[best_idx]
            for i in range(self.population_size):
                current_fox = self.population[i]
                if np.random.rand() < 0.5:
                    new_fox = self.jump_experience(current_fox)
                else:
                    new_fox = self.follow_leader(current_fox, best_fox)
                new_fitness = self.fitness(new_fox)
                if new_fitness > self.fitness(current_fox):
                    self.population[i] = new_fox
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_fox = new_fox.copy()
            print(f"PFO iteration best fitness: {best_fitness:.4f}")
        return best_fox

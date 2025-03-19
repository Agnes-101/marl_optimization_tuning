# algorithms/ga.py
import numpy as np
import random
from envs.custom_channel_env import evaluate_solution

class GAOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, generations=50, mutation_rate=0.1):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = [np.random.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
    def fitness(self, solution):
        return evaluate_solution(self.env, solution)["fitness"]
    
    def tournament_selection(self, k=3):
        participants = random.sample(self.population, k)
        return max(participants, key=self.fitness)
    
    def crossover(self, parent1, parent2):
        point = np.random.randint(1, self.num_users)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutate(self, solution):
        for i in range(self.num_users):
            if np.random.rand() < self.mutation_rate:
                solution[i] = np.random.randint(0, self.num_cells)
        return solution
    
    def optimize(self):
        for _ in range(self.generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2))
            self.population = new_population
        return max(self.population, key=self.fitness)

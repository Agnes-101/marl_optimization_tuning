# # algorithms/pfo.py

import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class PolarFoxOptimization:
    def __init__(self, num_users, num_cells, env, 
                 population_size=40, iterations=100,
                 mutation_factor=0.2, jump_rate=0.2, follow_rate=0.3, seed=42):
        np.random.seed(seed)
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_factor = mutation_factor
        self.jump_rate = jump_rate  
        self.follow_rate = follow_rate  
        self.seed = seed
        
        # Group parameters [PF, LF, a, b, m]
        self.types = np.array([
            [2, 2, 0.9, 0.9, 0.1],   # Group 0: Balanced
            [10, 2, 0.2, 0.9, 0.3],  # Group 1: Explorer
            [2, 10, 0.9, 0.2, 0.3],  # Group 2: Follower
            [2, 12, 0.9, 0.9, 0.01]  # Group 3: Conservative
        ])
        
        # Initialize population with groups
        self.population = self.initialize_population()
        self.group_weights = [1000, 1000, 1000, 1000]
        self.best_solution = None
        self.best_fitness = -np.inf

    def initialize_population(self):
        """Generate initial population with 20% heuristic solutions."""
        population = []
        for _ in range(self.population_size):
            if np.random.rand() < 0.4: # Before 0.2
                # Heuristic: Assign users to nearest cell
                fox = np.array([self.find_nearest_cell(pos) 
                              for pos in self.env.user_positions])
            else:
                # Random initialization
                fox = np.random.randint(0, self.num_cells, size=self.num_users)
            population.append(fox)
        return population

    def find_nearest_cell(self, user_position):
        """Find the nearest cell for a user position."""
        cell_positions = np.vstack([self.env.macro_positions, 
                                   self.env.small_positions])
        distances = np.linalg.norm(cell_positions - user_position, axis=1)
        return np.argmin(distances)

    def calculate_group_distribution(self):
        """Distribute foxes according to initial group ratios"""
        base_dist = np.array([0.25, 0.25, 0.25, 0.25])
        counts = np.floor(base_dist * self.population_size).astype(int)
        counts[3] = self.population_size - sum(counts[:3])
        return counts

    def create_fox(self, group_id):
        """Create fox with group-specific parameters"""
        fox = {
            "solution": self.generate_initial_solution(),
            "PF": self.types[group_id, 0],
            "LF": self.types[group_id, 1],
            "a": self.types[group_id, 2],
            "b": self.types[group_id, 3],
            "m": self.types[group_id, 4],
            "group": group_id
        }
        return fox

    # algorithms/pfo.py (modified)
    def generate_initial_solution(self):
        """20% use nearest-cell heuristic"""
        if np.random.rand() < 0.2:
            # Replace self.env.users with self.env.user_positions
            return np.array([self.env.find_nearest_cell(pos) for pos in self.env.user_positions])
        else:
            return np.random.randint(0, self.num_cells, size=self.num_users)

    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]

    def adaptive_parameters(self, iteration):
        """Decay jump power (a), boost follow power (b) over time"""
        decay_factor = 1 - (iteration / self.iterations) ** 0.5
        for i in range(4):
            self.types[i, 2] = max(self.types[i, 2] * decay_factor, 0.1)
            self.types[i, 3] = min(self.types[i, 3] / decay_factor, 0.99)

    def jump_experience(self, fox):
        """Randomly jump to explore new solutions."""
        new_fox = fox.copy()
        num_jumps = int(self.num_users * self.jump_rate)  # Convert to integer
        if num_jumps < 1:  # Ensure at least 1 jump
            num_jumps = 1
        indices = np.random.choice(self.num_users, num_jumps, replace=False)
        new_fox[indices] = np.random.randint(0, self.num_cells, size=num_jumps)
        return new_fox

    def follow_leader(self, current_fox, best_fox):
        """Update part of the solution to follow the best solution."""
        new_fox = current_fox.copy()
        num_follow = int(self.num_users * self.follow_rate)  # <-- FIXED
        indices = np.random.choice(self.num_users, num_follow, replace=False)
        new_fox[indices] = best_fox[indices]
        return new_fox

    def repair(self, solution):
        """Ensure cell capacity constraints"""
        cell_counts = np.bincount(solution, minlength=self.num_cells)
        overloaded = np.where(cell_counts > self.env.cell_capacity)[0]
        
        for cell in overloaded:
            users = np.where(solution == cell)[0]
            for user in users[self.env.cell_capacity:]:
                solution[user] = np.argmin(cell_counts)
                cell_counts[solution[user]] += 1
        return solution

    def leader_motivation(self, stagnation_count):
        """Reset underperforming foxes and adjust groups"""
        num_mutation = int(self.population_size * self.mutation_factor)
        for i in range(num_mutation):
            group_id = np.random.choice(4, p=self.group_weights/np.sum(self.group_weights))
            self.population[i] = self.create_fox(group_id)
        
        # Boost weights of best-performing group
        if self.best_solution is not None:
            best_group = self.population[np.argmax([f["group"] for f in self.population])]["group"]
            self.group_weights[best_group] += stagnation_count * 100

    def optimize(self):
        """Enhanced optimization loop with anti-stagnation mechanisms."""
        best_solution = self.population[0].copy()
        best_fitness = -np.inf
        historical_bests = []
        no_improvement_streak = 0
        stagnation_threshold = 5  # Increased from 3
        diversity_window = 10  # Track diversity over last N iterations
        mutation_reset = 0.1  # Minimum mutation factor
        
        # Initialize population diversity tracking
        diversity_history = []

        for iteration in range(self.iterations):
            # Evaluate population
            fitness_values = [self.fitness(fox) for fox in self.population]
            
            # Update best solution with elitism
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            # Maintain diversity tracking
            diversity = np.std(fitness_values)
            diversity_history.append(diversity)
            if len(diversity_history) > diversity_window:
                diversity_history.pop(0)

            # Update best solution with momentum (prevent oscillation)
            if current_best_fitness > best_fitness * 1.001:  # 0.1% improvement threshold
                best_fitness = current_best_fitness
                best_solution = self.population[current_best_idx].copy()
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1

            # Enhanced stagnation detection with diversity check
            avg_diversity = np.mean(diversity_history[-3:]) if diversity_history else 0
            if (iteration > 20 and 
                no_improvement_streak >= stagnation_threshold and 
                avg_diversity < 0.5 * np.mean(diversity_history)):
                
                # Aggressive mutation boost
                self.mutation_factor = min(1.0, self.mutation_factor * 2)
                no_improvement_streak = max(0, no_improvement_streak - 2)
                
                # Diversity injection
                num_replace = int(0.2 * self.population_size)
                for i in range(num_replace):
                    self.population[-(i+1)] = self.random_solution()
                
                print(f"Iter {iteration}: Mutation ↑ {self.mutation_factor:.2f}, Diversity injection")

            # Dynamic population management
            sorted_indices = np.argsort(fitness_values)[::-1]
            
            # Keep top 10% elites unchanged
            elite_count = max(1, int(0.1 * self.population_size))
            elites = [self.population[i].copy() for i in sorted_indices[:elite_count]]
            
            # Generate new population
            new_population = elites.copy()
            
            # Create remaining population through enhanced operations
            while len(new_population) < self.population_size:
                parent = self.population[np.random.choice(sorted_indices[:elite_count*2])]
                
                if np.random.rand() < 0.7:  # Favor exploitation
                    child = self.follow_leader(parent, best_solution)
                else:  # Exploration
                    child = self.jump_experience(parent)
                    
                # Apply mutation with adaptive probability
                mutation_prob = 0.3 + (0.5 * (self.mutation_factor / 1.0))
                if np.random.rand() < mutation_prob:
                    child = self.jump_experience(child)
                    
                new_population.append(child)

            self.population = new_population
            
            # Adaptive mutation decay
            if no_improvement_streak == 0 and self.mutation_factor > mutation_reset:
                self.mutation_factor *= 0.95  # Gradual decay on improvement

            # Progress tracking
            historical_bests.append(best_fitness)
            print(f"Iter {iteration+1}: Best = {best_fitness:.4f}, "
                f"Mutation = {self.mutation_factor:.2f}, "
                f"Diversity = {diversity:.2f}")

        return best_solution
    
    # def optimize(self):
    #     """Core optimization loop (fixed)."""
    #     best_solution = self.population[0].copy()
    #     best_fitness = self.fitness(best_solution)
    #     best_fitness = -np.inf  # Track the best fitness across iterations
    #     no_improvement_streak = 0  # Counter for consecutive iterations without improvement
    #     stagnation_threshold = 3  # Number of stagnant iterations before increasing mutation
                
    #     for iteration in range(self.iterations):
    #         # Evaluate population (no "solution" key needed)
    #         fitness_values = [self.fitness(fox) for fox in self.population]

    #         # Update best solution
    #         current_best_idx = np.argmax(fitness_values)
    #         current_best_fox = self.population[current_best_idx].copy()
    #         current_best_fitness = fitness_values[current_best_idx]
            
    #         # Update best solution
    #         if current_best_fitness > best_fitness:
    #             best_fitness = current_best_fitness
    #             best_solution = self.population[np.argmax(fitness_values)].copy()
    #             no_improvement_streak = 0  # Reset counter on improvement
    #         else:
    #             no_improvement_streak += 1  # Increment counter
    #                     # Dynamic mutation adjustmenhg
    #                     # t
    #         if iteration > 10 and no_improvement_streak >= stagnation_threshold:
    #             self.mutation_factor = min(0.5, self.mutation_factor + 0.15)
    #             no_improvement_streak = 0  # Reset after adjustment
    #             print(f"Iter {iteration}: Increased mutation to {self.mutation_factor:.2f}")
                
    #         # Mutation phase (using updated mutation_factor)
    #         for i in range(int(self.mutation_factor * self.population_size)):
    #             self.population[i] = self.jump_experience(self.population[i])
            
    #         # Generate new population
            
    #         new_population = []
    #         for fox in self.population:
    #             # Apply jump/follow directly to numpy arrays
    #             if np.random.rand() < 0.5:
    #                 new_fox = self.jump_experience(fox)
    #             else:
    #                 new_fox = self.follow_leader(fox, best_solution)

    #             # Replace if improved
    #             if self.fitness(new_fox) > self.fitness(fox):
    #                 new_population.append(new_fox)
    #             else:
    #                 new_population.append(fox)

    #         self.population = new_population
    #         print(f"Iter {iteration+1}: Best Fitness = {best_fitness:.4f}")

    #     return best_solution


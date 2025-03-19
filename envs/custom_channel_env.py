# File: envs/custom_env.py
import numpy as np
import gym
from gym import spaces

# -------------------------
# MobilityModel
# -------------------------
class MobilityModel:
    def __init__(self, area_size=100, mu_V=1.0, sigma_V=0.5, V_min=0.1, V_max=3.0, pause_prob=0.1):
        self.area_size = area_size
        self.mu_V = mu_V
        self.sigma_V = sigma_V
        self.V_min = V_min
        self.V_max = V_max
        self.pause_prob = pause_prob

    def update_positions(self, positions):
        num_users = positions.shape[0]
        pause_mask = np.random.rand(num_users) >= self.pause_prob
        V = np.clip(np.random.normal(self.mu_V, self.sigma_V, num_users), self.V_min, self.V_max)
        theta = np.random.uniform(0, 2 * np.pi, num_users)
        dx = V * np.cos(theta)
        dy = V * np.sin(theta)
        new_positions = positions.copy()
        new_positions[pause_mask, 0] = np.clip(positions[pause_mask, 0] + dx[pause_mask], 0, self.area_size)
        new_positions[pause_mask, 1] = np.clip(positions[pause_mask, 1] + dy[pause_mask], 0, self.area_size)
        return new_positions

# -------------------------
# ChannelModel
# -------------------------
class ChannelModel:
    def __init__(self, PL0=0, sigma_shadow=2.0,
                 beam_gain_main=10, beam_gain_side=1, beamwidth=np.deg2rad(30),
                 blockage_prob=0.3, NLOS_loss_dB=10, frequency=28e9):
        self.PL0 = PL0
        self.sigma_shadow = sigma_shadow
        self.beam_gain_main = beam_gain_main
        self.beam_gain_side = beam_gain_side
        self.beamwidth = beamwidth
        self.blockage_prob = blockage_prob
        self.NLOS_loss_linear = 10 ** (-NLOS_loss_dB / 10)
        self.frequency = frequency

    def get_path_loss(self, d):
        shadowing = np.random.normal(0, self.sigma_shadow)
        c = 3e8
        friis_loss = 20 * np.log10(4 * np.pi * d * self.frequency / c)
        PL_dB = self.PL0 + friis_loss + shadowing
        return 10 ** (-PL_dB / 10)

    def get_beamforming_gain(self, theta_diff):
        return self.beam_gain_main if abs(theta_diff) <= self.beamwidth / 2 else self.beam_gain_side

    def get_blockage_factor(self):
        return self.NLOS_loss_linear if np.random.rand() < self.blockage_prob else 1.0

    def calculate_channel_gain(self, d, theta_diff=0):
        pl = self.get_path_loss(d)
        bf_gain = self.get_beamforming_gain(theta_diff)
        blockage = self.get_blockage_factor()
        fading = max(0.5, np.random.rayleigh(scale=1.0))
        return pl * bf_gain * blockage * fading

# -------------------------
# Cell Class
# -------------------------
class Cell:
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = np.array(position)
        if cell_type == 'macro':
            self.transmit_power_dbm = 60
            self.path_loss_exponent = 3.5
        elif cell_type == 'small':
            self.transmit_power_dbm = 40
            self.path_loss_exponent = 2.5

# -------------------------
# Best Base Station Selection Function
# -------------------------
def select_best_bs(user_idx, env, bs_loads, base_penalty=20.0):
    # Compute SINR values for each base station.
    sinr_list = [env.calculate_sinr(user_idx, bs_idx) for bs_idx in range(env.num_cells)]
    # Compute distances from the user to each BS.
    distances = [np.linalg.norm(env.user_positions[user_idx] - env.cells[bs_idx].position)
                 for bs_idx in range(env.num_cells)]
    max_distance = max(distances) if max(distances) > 0 else 1
    max_sinr = max(sinr_list) if max(sinr_list) > 0 else 1
    norm_sinr = [sinr / max_sinr for sinr in sinr_list]
    max_load = max(bs_loads) if max(bs_loads) > 0 else 1
    load_penalty = [base_penalty * ((load / (max_load + 1e-6)) ** 2) for load in bs_loads]
    distance_penalty = [1.5 * (d / max_distance) for d in distances]
    sinr_spread_penalty = np.var(sinr_list) * 0.5

    selection_score = []
    for i in range(env.num_cells):
        if distances[i] > 50:  # Limit maximum association distance.
            score = -np.inf
        else:
            score = (0.7 * norm_sinr[i]) - (0.2 * load_penalty[i]) - (0.1 * distance_penalty[i]) - sinr_spread_penalty
        selection_score.append(score)
    
    best_bs_candidate = np.argmax(selection_score)
    if sinr_list[best_bs_candidate] < 0.1 * max_sinr:
        best_bs_candidate = np.argmax(sinr_list)
    
    current_bs = env.user_cell_assignment[user_idx]
    if current_bs != -1:
        current_sinr = env.calculate_sinr(user_idx, current_bs)
        candidate_sinr = env.calculate_sinr(user_idx, best_bs_candidate)
        if candidate_sinr < current_sinr * 1.10:
            return current_bs
    return best_bs_candidate

# -------------------------
# HeterogeneousEnvironment
# -------------------------
class HeterogeneousEnvironment:
    def __init__(self, macro_positions, small_positions, num_users, cell_capacity):
        self.cells = [Cell('macro', pos) for pos in macro_positions] + \
                     [Cell('small', pos) for pos in small_positions]
        self.num_cells = len(self.cells)
        self.num_users = num_users
        self.cell_capacity = cell_capacity
        x_positions = np.linspace(10, 90, num_users)
        y_positions = np.random.rand(num_users) * 100
        self.user_positions = np.column_stack((x_positions, y_positions))
        self.user_cell_assignment = np.full(num_users, -1)
        self.noise_power = 1e-9
        self.mobility_model = MobilityModel(area_size=100)
        self.channel_model = ChannelModel()
        self.current_step = 0

    def reset(self):
        x_positions = np.linspace(10, 90, self.num_users)
        y_positions = np.random.rand(self.num_users) * 100
        self.user_positions = np.column_stack((x_positions, y_positions))
        self.user_cell_assignment = np.full(self.num_users, -1)
        self.current_step = 0
        state = {
            "user_positions": self.user_positions,
            "cell_load": np.zeros(self.num_cells),
            "fairness": 1.0
        }
        return state

    def update_user_positions(self):
        old_positions = self.user_positions.copy()
        new_positions = self.mobility_model.update_positions(old_positions)
        if np.allclose(new_positions, old_positions):
            movement = np.random.uniform(-1, 1, old_positions.shape)
            new_positions = np.clip(old_positions + movement, 0, self.mobility_model.area_size)
        movement_magnitude = np.linalg.norm(new_positions - old_positions, axis=1)
        significant_movement = movement_magnitude > 3
        for i, moved in enumerate(significant_movement):
            if moved:
                self.user_cell_assignment[i] = -1  # Reset BS assignment if significant movement.
        self.user_positions = new_positions

    def fairness_index(self, bs_loads):
        total = np.sum(bs_loads)
        squared_sum = np.sum(bs_loads ** 2)
        return (total ** 2) / (self.num_cells * squared_sum) if squared_sum > 0 else 1.0

    def apply_actions(self, actions):
        # Update BS assignments using the provided solution.
        for user_idx, new_bs in enumerate(actions):
            self.user_cell_assignment[user_idx] = new_bs

    def calculate_sinr(self, user_idx, cell_idx):
        cell = self.cells[cell_idx]
        d = np.linalg.norm(self.user_positions[user_idx] - cell.position)
        channel_gain = max(1e-6, self.channel_model.calculate_channel_gain(d, theta_diff=0))
        G_tx = 10
        G_rx = 10
        SIR_target = 5
        P_min = 0.01
        P_tx = max(0.01, (P_min * (d ** cell.path_loss_exponent) * SIR_target) / (G_tx * G_rx * channel_gain))
        P_tx = min(P_tx, 10)
        signal_power = P_tx * channel_gain

        bs_positions = np.array([bs.position for bs in self.cells])
        distances = np.linalg.norm(self.user_positions[user_idx] - bs_positions, axis=1)
        bs_power_factors = np.array([min(10 ** ((bs.transmit_power_dbm - 30) / 10), 1.0) for bs in self.cells])
        gains = np.array([self.channel_model.calculate_channel_gain(dist, theta_diff=0) for dist in distances])
        gains[cell_idx] = 0.0
        inter_bs_interference = np.sum(bs_power_factors * gains)
        num_users_in_bs = np.sum(self.user_cell_assignment == cell_idx)
        intra_bs_interference = max(0.01, np.log1p(num_users_in_bs) * signal_power * 0.03)
        return signal_power / (inter_bs_interference + intra_bs_interference + self.noise_power)

    def step(self, actions=None):
        # Update positions first.
        self.update_user_positions()
        bs_loads = np.zeros(self.num_cells)
        if actions is None:
            # Instead of random assignment, use our best BS selection function.
            actions = []
            for user_idx in range(self.num_users):
                best_bs = select_best_bs(user_idx, self, bs_loads, base_penalty=20.0)
                actions.append(best_bs)
                bs_loads[best_bs] += 1
        else:
            for user_idx, new_bs in enumerate(actions):
                bs_loads[new_bs] += 1

        self.apply_actions(actions)
        sinr_list = [self.calculate_sinr(i, self.user_cell_assignment[i]) for i in range(self.num_users)]
        fairness = self.fairness_index(bs_loads)
        self.current_step += 1
        state = {
            "user_positions": self.user_positions,
            "cell_load": bs_loads,
            "fairness": fairness
        }
        return sinr_list, state

# -------------------------
# Evaluation Function (Detailed)
# -------------------------
def evaluate_solution(env, solution, alpha=0.1, beta=0.1):
    """
    Evaluate a candidate solution by applying BS assignments and computing multiple metrics:
      - Fitness: Sum of per-user rewards.
      - Each user's reward is defined as:
            log(1+SINR) - alpha*(distance/d_max) - beta*(load imbalance penalty)
    """
    env.apply_actions(solution)
    bs_loads = np.zeros(env.num_cells)
    for bs in solution:
        bs_loads[bs] += 1
    d_max = np.linalg.norm([env.mobility_model.area_size, env.mobility_model.area_size])
    
    rewards = []
    sinr_list = []
    throughput_list = []
    distances = []
    
    for user_idx in range(env.num_users):
        sinr = env.calculate_sinr(user_idx, solution[user_idx])
        sinr_list.append(sinr)
        throughput = np.log2(1 + sinr)
        throughput_list.append(throughput)
        bs_position = env.cells[solution[user_idx]].position
        user_position = env.user_positions[user_idx]
        distance = np.linalg.norm(user_position - bs_position)
        distances.append(distance)
        distance_penalty = distance / d_max
        load_penalty = abs(bs_loads[solution[user_idx]] - np.mean(bs_loads)) / (np.mean(bs_loads) + 1e-6)
        user_reward = np.log(1 + sinr) - alpha * distance_penalty - beta * load_penalty
        rewards.append(user_reward)
    
    fitness_value = np.sum(rewards)
    average_sinr = np.mean(sinr_list)
    average_throughput = np.mean(throughput_list)
    fairness = env.fairness_index(bs_loads)
    load_variance = np.var(bs_loads)
    
    return {
        "fitness": fitness_value,
        "average_sinr": average_sinr,
        "average_throughput": average_throughput,
        "fairness": fairness,
        "load_variance": load_variance,
        "rewards": rewards,
        "sinr_list": sinr_list,
        "bs_loads": bs_loads
    }

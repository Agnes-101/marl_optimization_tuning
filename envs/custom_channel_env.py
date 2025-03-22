import numpy as np
import gym
from gym import spaces

# ---------------------------------------------------
# MobilityModel
# ---------------------------------------------------
class MobilityModel:
    """
    Modified Random Waypoint (MRWP) model to update user positions.
    """
    def __init__(self, area_size=100, mu_V=1.0, sigma_V=0.5, V_min=0.1, V_max=3.0, 
                 pause_prob=0.1, rng=None):
        self.area_size = area_size
        self.mu_V = mu_V
        self.sigma_V = sigma_V
        self.V_min = V_min
        self.V_max = V_max
        self.pause_prob = pause_prob
        self.rng = rng if rng is not None else np.random.RandomState()

    def seed(self, seed=None):
        self.rng.seed(seed)
        return [seed]

    def update_positions(self, positions):
        num_users = positions.shape[0]
        pause_mask = self.rng.rand(num_users) >= self.pause_prob
        V = np.clip(self.rng.normal(self.mu_V, self.sigma_V, num_users), 
                    self.V_min, self.V_max)
        theta = self.rng.uniform(0, 2 * np.pi, num_users)
        dx = V * np.cos(theta)
        dy = V * np.sin(theta)
        new_positions = positions.copy()
        new_positions[pause_mask, 0] = np.clip(positions[pause_mask, 0] + dx[pause_mask], 0, self.area_size)
        new_positions[pause_mask, 1] = np.clip(positions[pause_mask, 1] + dy[pause_mask], 0, self.area_size)
        return new_positions

# ---------------------------------------------------
# ChannelModel
# ---------------------------------------------------
class ChannelModel:
    """
    Computes the channel gain using Friis free-space loss (with log-normal shadowing),
    directional beamforming gain, blockage probability, and Rayleigh fading.
    """
    def __init__(self, PL0=0, sigma_shadow=2.0,
                 beam_gain_main=10, beam_gain_side=1, beamwidth=np.deg2rad(30),
                 blockage_prob=0.3, NLOS_loss_dB=10, frequency=28e9, rng=None):
        self.PL0 = PL0
        self.sigma_shadow = sigma_shadow
        self.beam_gain_main = beam_gain_main
        self.beam_gain_side = beam_gain_side
        self.beamwidth = beamwidth
        self.blockage_prob = blockage_prob
        self.NLOS_loss_linear = 10 ** (-NLOS_loss_dB / 10)
        self.frequency = frequency
        self.bs_beam_directions = {}  # {bs_idx: current_beam_angle}
        self.rng = rng if rng is not None else np.random.RandomState()

    def seed(self, seed=None):
        self.rng.seed(seed)
        return [seed]

    def get_path_loss(self, d):
        shadowing = self.rng.normal(0, self.sigma_shadow)
        c = 3e8  # speed of light
        friis_loss = 20 * np.log10(4 * np.pi * d * self.frequency / c)
        PL_dB = self.PL0 + friis_loss + shadowing
        return 10 ** (-PL_dB / 10)

    def get_beamforming_gain(self, theta_diff):
        return self.beam_gain_main if abs(theta_diff) <= self.beamwidth / 2 else self.beam_gain_side

    def get_blockage_factor(self):
        return self.NLOS_loss_linear if self.rng.rand() < self.blockage_prob else 1.0
    
    def update_beam_directions(self, bs_positions, user_positions):
        """Dynamic beam alignment based on user positions"""
        for bs_idx, bs_pos in enumerate(bs_positions):
            user_vectors = user_positions - bs_pos
            if len(user_vectors) > 0:
                avg_angle = np.arctan2(np.mean(user_vectors[:, 1]), np.mean(user_vectors[:, 0]))
                self.bs_beam_directions[bs_idx] = avg_angle
                
    def calculate_channel_gain(self, d, bs_idx, user_pos, bs_pos, theta_diff=0):
        # Calculate angle difference from BS perspective
        user_angle = np.arctan2(user_pos[1] - bs_pos[1], user_pos[0] - bs_pos[0])
        theta_diff = user_angle - self.bs_beam_directions.get(bs_idx, 0)
        
        pl = self.get_path_loss(d)
        bf_gain = self.get_beamforming_gain(theta_diff)
        blockage = self.get_blockage_factor()
        # Use the seeded RNG for Rayleigh fading
        fading = max(0.5, self.rng.rayleigh(scale=1.0))
        return pl * bf_gain * blockage * fading

# ---------------------------------------------------
# Cell Class
# ---------------------------------------------------
class Cell:
    """
    Represents a base station. Macro and small cells are defined with different
    transmit powers and path loss exponents.
    """
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = np.array(position)
        if cell_type == 'macro':
            self.transmit_power_dbm = 60
            self.path_loss_exponent = 3.5
        elif cell_type == 'small':
            self.transmit_power_dbm = 40
            self.path_loss_exponent = 2.5

# ---------------------------------------------------
# Best Base Station Selection Function
# ---------------------------------------------------
def select_best_bs(user_idx, env, bs_loads, base_penalty=20.0):
    """
    Selects the best base station for a given user based on a combination of SINR,
    distance, and load penalties.
    """
    sinr_list = [env.calculate_sinr(user_idx, bs_idx) for bs_idx in range(env.num_cells)]
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
        if distances[i] > 50:
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

# ---------------------------------------------------
# HeterogeneousEnvironment
# ---------------------------------------------------
class HeterogeneousEnvironment:
    """
    The underlying simulation environment for a multi-tier cellular network.
    It handles user positions, channel effects, and base station assignments.
    """
    def __init__(self, macro_positions, small_positions, num_users, cell_capacity, seed=None):
        self.rng = np.random.RandomState(seed)
        self.cells = [Cell('macro', pos) for pos in macro_positions] + \
                     [Cell('small', pos) for pos in small_positions]
        self.num_cells = len(self.cells)
        self.num_users = num_users
        self.cell_capacity = cell_capacity
        self.user_positions = self.rng.uniform(0, 100, (num_users, 2))
        self.user_cell_assignment = np.full(num_users, -1)
        self.bs_loads = np.zeros(len(self.cells), dtype=int)
        bandwidth = 100e6
        noise_power_dBm = -174 + 10 * np.log10(bandwidth) + 7
        self.noise_power = 10 ** ((noise_power_dBm - 30) / 10)
        self.mobility_model = MobilityModel(rng=self.rng)
        self.channel_model = ChannelModel(rng=self.rng)
        self.current_step = 0

    def seed(self, seed=None):
        self.rng.seed(seed)
        self.mobility_model.seed(seed)
        self.channel_model.seed(seed)
        return [seed]

    def reset(self):
        self.user_positions = self.rng.uniform(0, 100, (self.num_users, 2))
        self.user_cell_assignment = np.full(self.num_users, -1)
        self.current_step = 0
        return {
            "user_positions": self.user_positions,
            "cell_load": np.zeros(self.num_cells),
            "fairness": 1.0
        }

    def update_user_positions(self):
        old_positions = self.user_positions.copy()
        new_positions = self.mobility_model.update_positions(old_positions)
        if np.allclose(new_positions, old_positions):
            # Use the seeded RNG for fallback movement
            movement = self.rng.uniform(-1, 1, old_positions.shape)
            new_positions = np.clip(old_positions + movement, 0, self.mobility_model.area_size)
        movement_magnitude = np.linalg.norm(new_positions - old_positions, axis=1)
        significant_movement = movement_magnitude > 3
        for i, moved in enumerate(significant_movement):
            if moved:
                self.user_cell_assignment[i] = -1
        self.user_positions = new_positions
        
    def fairness_index(self, throughputs):
        """Standardized throughput fairness"""
        return (np.sum(throughputs) ** 2) / (len(throughputs) * np.sum(np.square(throughputs)))

    def apply_actions(self, actions):
        """Enforce cell capacity constraints"""
        new_loads = np.bincount(actions, minlength=self.num_cells)
        
        for user_idx, new_bs in enumerate(actions):
            if new_loads[new_bs] <= self.cell_capacity:
                self.user_cell_assignment[user_idx] = new_bs
            else:
                least_loaded = np.argmin(self.bs_loads)
                self.user_cell_assignment[user_idx] = least_loaded
                self.bs_loads[least_loaded] += 1
        self.bs_loads = np.bincount(self.user_cell_assignment, minlength=self.num_cells)
    
    def _calculate_interference(self, user_idx, serving_cell_idx):
        """Calculate interference from all non-serving base stations."""
        user_pos = self.user_positions[user_idx]
        inter_interference = 0.0
        
        for bs_idx, cell in enumerate(self.cells):
            if bs_idx == serving_cell_idx:
                continue
            
            d = np.linalg.norm(user_pos - cell.position)
            beam_angle = self.channel_model.bs_beam_directions.get(bs_idx, 0)
            user_angle = np.arctan2(user_pos[1] - cell.position[1], user_pos[0] - cell.position[0])
            theta_diff = user_angle - beam_angle
            channel_gain = self.channel_model.calculate_channel_gain(
                d=d,
                bs_idx=bs_idx,
                user_pos=user_pos,
                bs_pos=cell.position,
                theta_diff=theta_diff
            )
            tx_power_watts = 10 ** ((cell.transmit_power_dbm - 30) / 10)
            inter_interference += tx_power_watts * channel_gain
            
        return inter_interference 
    
    def calculate_sinr(self, user_idx, cell_idx):
        cell = self.cells[cell_idx]
        user_pos = self.user_positions[user_idx]
        bs_pos = cell.position
        d = np.linalg.norm(user_pos - bs_pos)
        beam_angle = self.channel_model.bs_beam_directions.get(cell_idx, 0)
        user_angle = np.arctan2(user_pos[1] - bs_pos[1], user_pos[0] - bs_pos[0])
        theta_diff = user_angle - beam_angle
        channel_gain = self.channel_model.calculate_channel_gain(
            d=d,
            bs_idx=cell_idx,
            user_pos=user_pos,
            bs_pos=bs_pos,
            theta_diff=theta_diff
        )
        tx_power_watts = 10 ** ((cell.transmit_power_dbm - 30) / 10)
        rx_power = tx_power_watts * channel_gain / (d ** cell.path_loss_exponent)
        inter_interference = self._calculate_interference(user_idx, cell_idx)
        intra_interference = rx_power / (self.bs_loads[cell_idx] + 1e-6)
        total_interference = inter_interference + intra_interference + self.noise_power
        return rx_power / total_interference

    def step(self, actions=None):
        self.channel_model.update_beam_directions(
            [bs.position for bs in self.cells],
            self.user_positions
        )
        bs_idx = 0
        beam_angle = self.channel_model.bs_beam_directions.get(bs_idx, 0)
        print(f"Beam direction for BS {bs_idx}: {beam_angle:.2f} radians")
        self.update_user_positions()
        bs_loads = np.zeros(self.num_cells)
        if actions is None:
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

# ---------------------------------------------------
# Evaluation Function (Detailed)
# ---------------------------------------------------
def evaluate_detailed_solution(env, solution, alpha=0.1, beta=0.1):
    """
    Evaluate a candidate solution by applying base station assignments
    and computing multiple metrics.
    """
    env.apply_actions(solution)
    bs_loads = np.zeros(env.num_cells)
    for bs in solution:
        bs_loads[bs] += 1
    
    d_max = env.mobility_model.area_size * np.sqrt(2)
    
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
        load_imbalance = np.std(bs_loads) / (np.mean(bs_loads) + 1e-6)
        user_reward = np.log(1 + sinr) - alpha * distance_penalty - beta * load_imbalance       
        rewards.append(user_reward)
    
    fitness_value = np.sum(rewards)
    average_sinr = np.mean(sinr_list)
    average_throughput = np.mean(throughput_list)
    fairness = (np.sum(throughput_list) ** 2) / (len(throughput_list) * np.sum(np.square(throughput_list)))
    load_variance = np.var(bs_loads)
    
    return {
        "fitness": fitness_value,
        "average_sinr": average_sinr,
        "average_throughput": average_throughput,
        "fairness": fairness,
        "load_variance": load_variance,
        "bs_loads": bs_loads
    }

# ---------------------------------------------------
# PyMARLCustomEnv (Gym-Compatible Wrapper)
# ---------------------------------------------------
class PyMARLCustomEnv(gym.Env):
    def __init__(self, env_args):
        super(PyMARLCustomEnv, self).__init__()
        self.num_users = env_args.get("num_users", 60)
        self.rng = np.random.RandomState(env_args.get("seed"))
        self.macro_positions = env_args.get("macro_positions", [[50, 50]])
        self.small_positions = env_args.get("small_positions", [[20, 20], [20, 80], [80, 20], [80, 80]])
        self.cell_capacity = env_args.get("cell_capacity", 15)
        self.episode_limit = env_args.get("episode_limit", 50)
        self.env = HeterogeneousEnvironment(
            self.macro_positions,
            self.small_positions,
            self.num_users,
            self.cell_capacity,
            seed=env_args.get("seed")
        )
        self.num_cells = self.env.num_cells
        self.obs_dim = 4  # [x_norm, y_norm, load_norm, sinr]
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cells)
        self.current_step = 0

    def seed(self, seed=None):
        self.rng.seed(seed)
        self.env.seed(seed)
        return [seed]
        
    def __getattr__(self, attr):
        return getattr(self.env, attr)
        
    def _normalize_position(self, pos):
        return pos / self.env.mobility_model.area_size
    
    def reset(self):
        state = self.env.reset()
        obs = []
        for i in range(self.num_users):
            assigned_bs = self.user_cell_assignment[i]
            load_value = 0
            if assigned_bs != -1:
                load_value = self.env.bs_loads[assigned_bs] / self.cell_capacity
            # Use 0 as default BS index if unassigned
            norm_sinr = self.env.calculate_sinr(i, assigned_bs if assigned_bs != -1 else 0) / 20
            obs_i = [
                *self._normalize_position(state["user_positions"][i]),
                load_value,
                norm_sinr
            ]
            obs.append(obs_i)
        global_state = np.concatenate(obs)
        self.current_step = 0
        return {"obs": obs, "state": global_state}

    def step(self, actions):
        sinr_list, state = self.env.step(actions)
        obs = []
        rewards = []
        for i in range(self.num_users):
            obs_i = np.concatenate([
                state["user_positions"][i],
                [state["fairness"]],
                state["cell_load"]
            ])
            obs.append(obs_i)
            rewards.append(np.log(1 + sinr_list[i]))
        global_state = np.concatenate(obs)
        self.current_step += 1
        done = self.current_step >= self.episode_limit
        return {"obs": obs, "state": global_state, "reward": rewards, "done": done, "info": {}}

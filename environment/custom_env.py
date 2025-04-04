# environment/custom_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RuralSolarMaintenanceEnv(gym.Env):
    def __init__(self, grid_size=3, max_episode_length=50):
        super(RuralSolarMaintenanceEnv, self).__init__()
        self.grid_size = grid_size
        self.max_episode_length = max_episode_length
        self.action_space = spaces.Discrete(6)  # 0: right, 1: left, 2: down, 3: up, 4: inspect, 5: maintain
        self.observation_space = spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32)

        # Initialize state
        self.robot_pos = [0, 0]
        self.robot_energy = 5.0
        self.panel_health = np.ones((grid_size, grid_size))
        self.panel_dust = np.zeros((grid_size, grid_size))
        self.battery_efficiency = np.ones((grid_size, grid_size))
        self.weather = 0  # 0: clear, 1: cloudy, 2: dusty
        self.total_power = 0.0
        self.step_count = 0
        self.weather_change_prob = 0.1
        self.degradation_rate = 0.01

    def _update_environment(self):
        # Update weather
        if np.random.random() < self.weather_change_prob:
            self.weather = np.random.choice([0, 1, 2])

        # Update panel dust and health
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Dust accumulation
                if self.weather == 1 and np.random.random() < 0.2:
                    self.panel_dust[i, j] = min(1.0, self.panel_dust[i, j] + 0.1)
                elif self.weather == 2 and np.random.random() < 0.3:
                    self.panel_dust[i, j] = min(1.0, self.panel_dust[i, j] + 0.15)

                # Health degradation
                if np.random.random() < self.degradation_rate:
                    self.panel_health[i, j] = max(0.0, self.panel_health[i, j] - 0.05)

                # Battery efficiency
                self.battery_efficiency[i, j] = self.panel_health[i, j] * (1 - self.panel_dust[i, j])
                if self.weather == 1:
                    self.battery_efficiency[i, j] *= 0.7
                elif self.weather == 2:
                    self.battery_efficiency[i, j] *= 0.5

        # Update total power
        self.total_power = np.mean(self.battery_efficiency)

    def step(self, action):
        self.step_count += 1

        # Update robot position based on action
        x, y = self.robot_pos
        if action == 0:  # Right
            y = min(self.grid_size - 1, y + 1)
            self.robot_energy -= 0.005
        elif action == 1:  # Left
            y = max(0, y - 1)
            self.robot_energy -= 0.005
        elif action == 2:  # Down
            x = min(self.grid_size - 1, x + 1)
            self.robot_energy -= 0.005
        elif action == 3:  # Up
            x = max(0, x - 1)
            self.robot_energy -= 0.005
        elif action == 4:  # Inspect
            self.robot_energy -= 0.01
        elif action == 5:  # Maintain
            self.panel_dust[x, y] = max(0.0, self.panel_dust[x, y] - 0.3)
            self.panel_health[x, y] = min(1.0, self.panel_health[x, y] + 0.2)
            self.robot_energy -= 0.02

        self.robot_pos = [x, y]
        print(f"Step - Action: {action}, New Robot Pos: {self.robot_pos}, Energy: {self.robot_energy}")  # Debug print

        # Update environment
        self._update_environment()

        # Check if done
        done = self.robot_energy <= 0
        truncated = self.step_count >= self.max_episode_length

        # Calculate reward
        reward = self.total_power - 0.1
        if action == 4:
            reward += 0.05
        elif action == 5:
            reward += 0.1

        # Get observation
        obs = self._get_obs()
        info = {}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = [0, 0]
        self.robot_energy = 5.0
        self.panel_health = np.ones((self.grid_size, self.grid_size))
        self.panel_dust = np.zeros((self.grid_size, self.grid_size))
        self.battery_efficiency = np.ones((self.grid_size, self.grid_size))
        self.weather = 0
        self.total_power = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(32, dtype=np.float32)
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                obs[idx] = self.panel_health[i, j]
                obs[idx + 1] = self.panel_dust[i, j]
                obs[idx + 2] = self.battery_efficiency[i, j]
                idx += 3
        obs[idx] = self.robot_pos[0] / self.grid_size
        obs[idx + 1] = self.robot_pos[1] / self.grid_size
        obs[idx + 2] = self.robot_energy / 5.0
        obs[idx + 3] = self.weather / 2.0
        return obs
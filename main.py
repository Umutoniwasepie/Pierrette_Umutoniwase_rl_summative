# main.py
import os
import torch
import random
from stable_baselines3 import PPO
from environment.custom_env import RuralSolarMaintenanceEnv
from environment.rendering import RuralSolarRenderer
from training.dqn_training import DQNAgent
import numpy as np
import imageio

def ensure_directories():
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

def create_visualizations():
    # Create environment and renderer
    env = RuralSolarMaintenanceEnv(grid_size=3, max_episode_length=50)
    renderer = RuralSolarRenderer(env, window_size=608)

    # Create static image
    renderer.render_static(output_path="visualizations/solar_farm_static.png")

    # Create GIF using the heuristic policy
    renderer.create_gif(output_path="visualizations/solar_farm.gif", num_frames=50)

    renderer.close()

def record_video():
    # Increase max_episode_length to 500 to allow videos to run for up to 50 seconds (at 10 FPS)
    env = RuralSolarMaintenanceEnv(grid_size=3, max_episode_length=500)
    renderer = RuralSolarRenderer(env, window_size=608)

    # Load DQN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQNAgent(state_dim=32, action_dim=6, device=device, lr=0.001, gamma=0.95, epsilon_decay=0.999, batch_size=128)
    dqn_agent.policy_net.load_state_dict(torch.load("models/dqn/dqn_best_model.pth", map_location=device))
    dqn_agent.policy_net.eval()

    # Load PPO with custom objects
    ppo_model = PPO.load("models/pg/ppo_best_model", custom_objects={"clip_range": 0.2, "lr_schedule": lambda x: 0.0003})

    exploration_prob = 0.2  # Probability of taking a random movement action

    # DQN Video with real-time rendering
    frames = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        # Define the episode function for DQN with exploration
        def dqn_episode_func(state):
            if random.random() < exploration_prob:
                action = random.choice([0, 1, 2, 3])  # Random movement action
            else:
                action = dqn_agent.act(state, explore=False)
            print(f"DQN Action: {action}, Robot Pos: {env.robot_pos}, Energy: {env.robot_energy}")
            return action

        # Capture frame and render dynamically
        image = renderer.render_static(output_path=None)
        frames.append(np.array(image))
        state, done, truncated = renderer.render_dynamic(episode_func=dqn_episode_func, max_steps=1)

    imageio.mimsave("visualizations/dqn_episode.mp4", frames, fps=10)
    print(f"DQN video saved to visualizations/dqn_episode.mp4 with {len(frames)} frames")

    # PPO Video with real-time rendering
    frames = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        # Define the episode function for PPO with exploration
        def ppo_episode_func(state):
            if random.random() < exploration_prob:
                action = random.choice([0, 1, 2, 3])  # Random movement action
            else:
                action, _ = ppo_model.predict(state)
            print(f"PPO Action: {action}, Robot Pos: {env.robot_pos}, Energy: {env.robot_energy}")
            return action

        # Capture frame and render dynamically
        image = renderer.render_static(output_path=None)
        frames.append(np.array(image))
        state, done, truncated = renderer.render_dynamic(episode_func=ppo_episode_func, max_steps=1)

    imageio.mimsave("visualizations/ppo_episode.mp4", frames, fps=10)
    print(f"PPO video saved to visualizations/ppo_episode.mp4 with {len(frames)} frames")

    renderer.close()

if __name__ == "__main__":
    ensure_directories()
    create_visualizations()
    record_video()
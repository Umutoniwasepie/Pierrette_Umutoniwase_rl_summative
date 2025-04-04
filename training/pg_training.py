# training/ppo_training.py
import sys
import os

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import RuralSolarMaintenanceEnv
from stable_baselines3.common.callbacks import BaseCallback

# Ensure the models/pg directory exists
os.makedirs("models/pg", exist_ok=True)

# Custom callback to log policy entropy
class EntropyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EntropyCallback, self).__init__(verbose)
        self.entropies = []
    
    def _on_step(self) -> bool:
        # Compute policy entropy
        if hasattr(self.model, 'policy'):
            obs = self.locals['obs_tensor']
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs)
                entropy = dist.entropy().mean().item()
            self.entropies.append(entropy)
        return True

def evaluate_agent(model, env, num_episodes=100, seed_offset=0):
    total_rewards = []
    steps_per_episode = []
    for i in range(num_episodes):
        env.seed(seed_offset + i)
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
        total_rewards.append(episode_reward)
        steps_per_episode.append(steps)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(steps_per_episode)
    return avg_reward, std_reward, avg_steps

def train_ppo():
    env = RuralSolarMaintenanceEnv(grid_size=3, max_episode_length=50)
    env = Monitor(env)
    
    hyperparams = [
        {"learning_rate": 0.0003, "n_steps": 2048, "batch_size": 64, "gamma": 0.99},
        {"learning_rate": 0.0001, "n_steps": 1024, "batch_size": 32, "gamma": 0.99},
        {"learning_rate": 0.0003, "n_steps": 2048, "batch_size": 64, "gamma": 0.95},
        {"learning_rate": 0.0005, "n_steps": 2048, "batch_size": 128, "gamma": 0.99},
    ]
    
    total_timesteps = 100000  # Increased from 50,000 to 100,000
    best_mean_reward = float('-inf')
    best_model = None
    best_config = None
    all_rewards = []
    all_entropies = []
    
    for config_idx, config in enumerate(hyperparams):
        print(f"\nTraining with configuration {config_idx + 1}/{len(hyperparams)}: {config}")
        
        # Create callback for entropy logging
        entropy_callback = EntropyCallback()
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            ent_coef=0.05,  # Increased from 0.01 to 0.05
            verbose=0,
            tensorboard_log="./ppo_tensorboard/"
        )
        
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=entropy_callback)
        
        mean_reward, std_reward, avg_steps = evaluate_agent(model, env, seed_offset=0)
        print(f"Config {config_idx + 1} Evaluation (Training Seeds) - Mean Reward: {mean_reward:.2f}, Std Dev: {std_reward:.2f}, Average Steps: {avg_steps:.2f}")
        
        # Evaluate on unseen seeds for generalization
        mean_reward_unseen, std_reward_unseen, avg_steps_unseen = evaluate_agent(model, env, seed_offset=1000)
        print(f"Config {config_idx + 1} Evaluation (Unseen Seeds) - Average Reward: {mean_reward_unseen:.2f}, Std Dev: {std_reward_unseen:.2f}, Average Steps: {avg_steps_unseen:.2f}")
        
        rewards = []
        for i in range(0, total_timesteps, 2048):
            mean_reward_intermediate, _, _ = evaluate_agent(model, env, num_episodes=10)
            rewards.append(mean_reward_intermediate)
        all_rewards.append(rewards)
        all_entropies.append(entropy_callback.entropies)
        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_model = model
            best_config = config
    
    # Save the best model
    best_model.save("models/pg/ppo_best_model")
    print(f"\nBest Configuration: {best_config}")
    print(f"Best Mean Reward: {best_mean_reward:.2f}")
    print("Best model saved as 'models/pg/ppo_best_model.zip'")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for idx, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f"Config {idx + 1}")
    plt.title("PPO Training Rewards")
    plt.xlabel("Evaluation Step (every 2048 timesteps)")
    plt.ylabel("Mean Reward")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for idx, entropies in enumerate(all_entropies):
        plt.plot(entropies, label=f"Config {idx + 1}")
    plt.title("PPO Policy Entropy")
    plt.xlabel("Training Step")
    plt.ylabel("Entropy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ppo_training_results.png")
    plt.show()

    return best_config, best_mean_reward

if __name__ == "__main__":
    train_ppo()
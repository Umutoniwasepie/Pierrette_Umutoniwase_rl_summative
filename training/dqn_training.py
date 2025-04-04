# training/dqn_training.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from environment.custom_env import RuralSolarMaintenanceEnv

# Ensure the models/dqn directory exists
os.makedirs("models/dqn", exist_ok=True)

# DQN Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, device, lr, gamma, epsilon_decay, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Main and target networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(50000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.target_update = 50  # Increased from 10 to 50
    
    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def train(self, episode, env, num_episodes):
        if len(self.memory) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)
        
        # Double DQN: Select actions using policy_net, evaluate using target_net
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

def evaluate_agent(agent, env, num_episodes=100, seed_offset=0):
    total_rewards = []
    steps_per_episode = []
    for i in range(num_episodes):
        # Use a different seed for each episode to test generalization
        env.seed(seed_offset + i)
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.act(state, explore=False)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        total_rewards.append(episode_reward)
        steps_per_episode.append(steps)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(steps_per_episode)
    return avg_reward, std_reward, avg_steps

def train_dqn():
    env = RuralSolarMaintenanceEnv(grid_size=3, max_episode_length=50)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hyperparams = [
        {"lr": 0.001, "gamma": 0.99, "epsilon_decay": 0.999, "batch_size": 128},
        {"lr": 0.0005, "gamma": 0.99, "epsilon_decay": 0.999, "batch_size": 64},
        {"lr": 0.001, "gamma": 0.95, "epsilon_decay": 0.999, "batch_size": 128},
        {"lr": 0.0001, "gamma": 0.99, "epsilon_decay": 0.999, "batch_size": 128},
    ]
    
    num_episodes = 1000
    best_avg_reward = float('-inf')
    best_model = None
    best_config = None
    all_rewards = []
    all_losses = []
    all_steps = []
    
    for config_idx, config in enumerate(hyperparams):
        print(f"\nTraining with configuration {config_idx + 1}/{len(hyperparams)}: {config}")
        agent = DQNAgent(state_dim, action_dim, device, **config)
        rewards = []
        losses = []
        steps_per_episode = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.memory.push(state, action, reward, next_state, done or truncated)
                state = next_state
                episode_reward += reward
                steps += 1
                
                loss = agent.train(episode, env, num_episodes)
                if loss is not None:
                    losses.append(loss)
            
            rewards.append(episode_reward)
            steps_per_episode.append(steps)
            
            if episode % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            if episode % 50 == 0:
                print(f"Config {config_idx + 1}, Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {steps}")
        
        all_rewards.append(rewards)
        all_losses.append(losses)
        all_steps.append(steps_per_episode)
        
        # Evaluate on training seeds
        avg_reward, std_reward, avg_steps = evaluate_agent(agent, env, seed_offset=0)
        print(f"Config {config_idx + 1} Evaluation (Training Seeds) - Average Reward: {avg_reward:.2f}, Std Dev: {std_reward:.2f}, Average Steps: {avg_steps:.2f}")
        
        # Evaluate on unseen seeds for generalization
        avg_reward_unseen, std_reward_unseen, avg_steps_unseen = evaluate_agent(agent, env, seed_offset=1000)
        print(f"Config {config_idx + 1} Evaluation (Unseen Seeds) - Average Reward: {avg_reward_unseen:.2f}, Std Dev: {std_reward_unseen:.2f}, Average Steps: {avg_steps_unseen:.2f}")
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model = agent.policy_net.state_dict()
            best_config = config
    
    # Save the best model
    torch.save(best_model, "models/dqn/dqn_best_model.pth")
    print(f"\nBest Configuration: {best_config}")
    print(f"Best Average Reward: {best_avg_reward:.2f}")
    print("Best model saved as 'models/dqn/dqn_best_model.pth'")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for idx, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f"Config {idx + 1}")
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for idx, losses in enumerate(all_losses):
        plt.plot(losses, label=f"Config {idx + 1}")
    plt.title("DQN Training Losses")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("dqn_training_results.png")
    plt.show()

    return best_config, best_avg_reward

if __name__ == "__main__":
    train_dqn()
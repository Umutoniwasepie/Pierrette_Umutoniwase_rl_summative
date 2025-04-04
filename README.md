# Pierrette_Umutoniwase_rl_summative

## Project Overview
This project focuses on developing a reinforcement learning (RL) environment to simulate the maintenance of a solar farm using an autonomous robot. The primary objective is to train an RL agent to optimize the maintenance of solar panels by managing their health, dust accumulation, and battery efficiency under varying weather conditions. The environment, named RuralSolarMaintenanceEnv, provides a grid-based simulation where the robot navigates to clean panels, repair damage, and ensure optimal energy production.

The project serves as an assessment for exploring RL techniques in a practical, simulated setting, with a focus on renewable energy maintenance.

## Environment
The RuralSolarMaintenanceEnv is a custom RL environment designed to simulate a solar farm maintenance scenario. Key features of the environment include:

- Grid-Based Layout: The solar farm is represented as a grid (default size: 3x3), where each cell contains a solar panel.
- Robot Agent: A single robot navigates the grid to perform maintenance tasks (e.g., cleaning dust, repairing panels).
  
**State Variables:**
 - Panel Health: Each panel has a health value (0 to 1), where lower values indicate damage.
 - Panel Dust: Dust accumulation on panels (0 to 1), affecting efficiency.
 - Battery Efficiency: Each panel has a battery with efficiency (0 to 1), impacting energy storage.
 - Robot Energy: The robot's energy level (0 to 1), which depletes with actions and can be recharged.
 - Weather: Three weather conditions (Clear, Cloudy, Dust Storm) that affect panel efficiency and dust accumulation.
 - Total Power: The overall power output of the solar farm, based on panel health, dust, and battery efficiency.

**Actions:**
  - Move (Up, Down, Left, Right)
  - Clean dust
  - Repair panel
  - Idle
  
**Reward System:**
  - Positive rewards for increasing total power output, improving battery level, and performing effective maintenance.
  - Negative rewards for unnecessary actions or excessive wear on panels.
    
**Termination Conditions:**
  - Episode ends after a fixed number of steps.
  - If all panels are at peak performance.

## Reinforcement Learning Algorithms
### 1. Deep Q-Network (DQN)
DQN is a value-based RL algorithm that uses a deep neural network to approximate the Q-values of state-action pairs.

**Hyperparameters Used:**
- Learning Rate: `0.001`
- Discount Factor (Gamma): `0.99`
- Replay Buffer Size: `50000`
- Batch Size: `128`
- epsilon_decay: `0.999`
- Exploration Strategy: `ε-greedy` (ε decays from 1.0 to 0.1)
- Target Network Update Frequency: `1000`

**Why These Hyperparameters?**
- A lower learning rate ensures stable updates.
- A high discount factor ensures long-term planning.
- A large replay buffer helps in better experience utilization.
- The exploration decay balances initial exploration with later exploitation.

### 2. Proximal Policy Optimization (PPO)
PPO is a policy-based RL algorithm that updates policies in a stable and efficient manner using clipped objective functions.

**Hyperparameters Used:**
- Learning Rate: `0.0003`
- Discount Factor (Gamma): `0.95`
- n_steps: `2048`
- PPO Clip Range: `0.2`
- ent_coef: `0.05`
- Batch Size: `64`
- Number of Epochs per Update: `10`

**Why These Hyperparameters?**
- PPO Clip Range (`0.2`) helps prevent large updates, stabilizing training.
- ent_coef (`0.05`) balances bias and variance in advantage estimation.
- Training epochs (`10`) ensure sufficient updates per batch.

## Results
- **DQN:** Converges to an optimal policy after ~10,000 episodes, achieving a high average reward.
- **PPO:** Shows faster convergence than DQN but requires careful tuning to prevent performance drops.
- **Comparison:** PPO outperforms DQN in terms of sample efficiency, but DQN achieves more stable final policies.

**Performance Metrics:**
- **Reward Progression:** (Graphs & comparisons)
- **Action Distribution:** (How often actions were taken)
- **Final Policy Visualization:** (Heatmap of panel conditions)

**GIF Visualization of Trained Agent:**
![Trained Agent GIF](visualizations/solar_farm.gif)

## How to Use
### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Run Environment Visualization
```bash
python main.py
```

### 3. Train DQN Agent
```bash
python training/dqn_training.py
```

### 4. Train PPO Agent
```bash
python training/pg_training.py
```

## Repository Structure
📂 `project_root/`  
├── 📂 `environment/` *(Custom Gym environment & PyOpenGL visualization)*  
├── 📂 `training/` *(DQN & PPO training scripts using SB3)*  
├── 📂 `models/` *(Saved models for DQN & PPO)*  
├── `main.py` *(Entry point for experiments)*  
├── `requirements.txt` *(Dependencies)*  
└── `README.md` *(Project documentation & GIF visualization)*  

## Future Work
- Improve the reward function to incorporate real-world power loss calculations.
- Add stochastic weather conditions to increase environment variability.
- Test with more advanced RL algorithms like SAC or TD3.

## References
1. OpenAI Gym Documentation
2. Stable Baselines3 for RL Training
3. Reinforcement Learning for Energy Systems Research



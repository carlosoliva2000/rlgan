import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
# import flappy_bird_gymnasium
import math
from itertools import count
from dqn.dqn_agent import DqnAgent
from dqn.dqn_memory import Transition, ReplayMemory
from dqn.dueling_dqn import DuelingNet
from dqn.simple_network import SimpleNet

# env = gym.make('FlappyBird-v0', render_mode=None)
env = gym.make("LunarLander-v2", render_mode='human')
input_size = env.observation_space.shape[0]
hidden_size = 64
output_size = env.action_space.n
device = 'cuda'

policy_net = DuelingNet(input_size, hidden_size, output_size).to(device)
policy_net.load_state_dict(torch.load('lunar_lander_dueling_softupdate.pt'))
policy_net.eval()  # Target net will not be trained

memory = ReplayMemory(10000)

batch_size = 128
gamma = 0.99  # Discount factor

# Hyperparameters
num_episodes = 10
epsilon_start = 0
epsilon_end = 0.0
epsilon_decay = 200
target_update = 4  # How often to update the target network


steps_done = 0

for episode in range(num_episodes):
    # Initialize the environment and state
    state, _ = env.reset()
    state = torch.tensor([state], dtype=torch.float).to(device)
    total_reward = 0
    for t in count():
        # Select and perform an action
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  math.exp(-1. * steps_done / epsilon_decay)
        action = DqnAgent.epsilon_greedy_policy(state, epsilon, policy_net, output_size, device)
        steps_done += 1

        next_state, reward, done, truncated, _ = env.step(action.item())
        # if action.item() == 0:
        #     reward -= 0.1
        total_reward += reward
        reward = torch.tensor([reward], dtype=torch.float).to(device)

        if t > 500:
            truncated = True

        if not done and not truncated:
            next_state = torch.tensor([next_state], dtype=torch.float).to(device)
        else:
            next_state = None

        # Move to the next state
        state = next_state

        if done or truncated:
            print(f"Episode {episode} finished after {t + 1} timesteps. Reward: {total_reward}. Epsilon: {epsilon}")
            break

print('Complete')
env.close()

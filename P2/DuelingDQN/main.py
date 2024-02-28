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
import pandas as pd

# env = gym.make('FlappyBird-v0', render_mode=None)
env = gym.make("LunarLander-v2", render_mode=None)
input_size = env.observation_space.shape[0]
hidden_size = 64
output_size = env.action_space.n
device = 'cuda'

policy_net = DuelingNet(input_size, hidden_size, output_size).to(device)
# policy_net.load_state_dict(torch.load('flappy_bird_88999.pt'))
target_net = DuelingNet(input_size, hidden_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target net will not be trained

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(100000)

batch_size = 64
gamma = 0.99  # Discount factor

# Hyperparameters
num_episodes = 500
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = 200
target_update = 1  # How often to update the target network

all_rewards = []
all_avg_rewards = []
all_steps = []
all_non_zero_actions = []
steps_done = 0

for episode in range(num_episodes):
    # Initialize the environment and state
    state, _ = env.reset()
    state = torch.tensor([state], dtype=torch.float).to(device)
    total_reward = 0
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
              math.exp(-1. * steps_done / epsilon_decay)
    total_non_zero_actions = 0
    for t in range(1001):

        # Select and perform an action
        # epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
        #           math.exp(-1. * steps_done / epsilon_decay)
        action = DqnAgent.epsilon_greedy_policy(state, epsilon, policy_net, output_size, device)
        steps_done += 1

        next_state, reward, done, truncated, _ = env.step(action.item())
        real_reward = reward

        if action.item() != 0:
            total_non_zero_actions += 1

        total_reward += real_reward
        reward = torch.tensor([reward], dtype=torch.float).to(device)

        if t >= 1000:
            truncated = True

        if not done and not truncated:
            next_state = torch.tensor([next_state], dtype=torch.float).to(device)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        DqnAgent.optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, device)

        if t % 120 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done or truncated:
            all_rewards.append(total_reward)
            avg_reward = np.mean(all_rewards)
            all_steps.append(t)
            all_avg_rewards.append(avg_reward)
            all_non_zero_actions.append(total_non_zero_actions)
            print(f"Episode {episode} finished after {t + 1} timesteps. Reward: {total_reward}. Epsilon: {epsilon}")
            break


# save the network
torch.save(policy_net.state_dict(), "lunar_lander_dueling_softupdate.pt")
print('Complete')
env.close()
# save all rewards as csv
final_data = {
    'rewards': all_rewards,
    'avg_rewards': all_avg_rewards,
    'steps': all_steps,
    'non_zero_actions': all_non_zero_actions
}
df = pd.DataFrame(final_data)
df.to_csv('lunar_lander_dueling_rewards_softupdate.csv')


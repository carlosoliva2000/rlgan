import os
import time
from matplotlib.pylab import f
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

from replay_buffer import ReplayBuffer


def DeepQNetwork(lr, num_actions, input_dims, hidden_dims, name=None) -> Sequential:
    q_net = Sequential(name=name)
    q_net.add(Dense(hidden_dims[0], input_dim=input_dims, activation='relu'))
    for dim in hidden_dims[1:]:
        q_net.add(Dense(dim, activation='relu'))
    q_net.add(Dense(num_actions, activation=None))

    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    q_net.summary()

    return q_net


class Agent:
    def __init__(self, lr, discount_factor, num_actions, batch_size, input_dims, hidden_dims, epsilon=1.0, epsilon_decay=0.995, epsilon_final=0.01, update_rate=120):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final
        self.update_rate = update_rate
        self.step_counter = 0
        self.buffer = ReplayBuffer(1000000, input_dims)
        self.q_net = DeepQNetwork(lr, num_actions, input_dims, hidden_dims, name='Main_Network')
        self.q_target_net = DeepQNetwork(lr, num_actions, input_dims, hidden_dims, name='Target_Network')


    def store_tuple(self, state, action, reward, new_state, done):
        self.buffer.store_tuples(state, action, reward, new_state, done)


    def policy(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_net(state, training=False)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action


    def train_step(self):
        if self.step_counter % self.update_rate == 0:  # Update the target network
            self.q_target_net.set_weights(self.q_net.get_weights())

        # Sample a batch from the buffer
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.buffer.sample_buffer(self.batch_size)

        # Predict the Q values for the current state (main network) and the next state (target network)
        q_predicted = self.q_net(state_batch, training=False)
        q_next = self.q_target_net(new_state_batch, training=False)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        # Update the Q values for the actions taken with Bellman's equation
        for idx in range(done_batch.size):
            q_target[idx, action_batch[idx]] = reward_batch[idx] + (1-done_batch[idx]) * self.discount_factor * q_max_next[idx]
        # for idx in range(done_batch.shape[0]):
        #     target_q_val = reward_batch[idx]
        #     if not done_batch[idx]:
        #         target_q_val += self.discount_factor*q_max_next[idx]
        #     q_target[idx, action_batch[idx]] = target_q_val

        # Train the main network with the updated Q values
        loss = self.q_net.train_on_batch(state_batch, q_target)
        # self.q_net.fit(state_batch, q_target, epochs=1, verbose=0, batch_size=self.batch_size)

        self.step_counter += 1
        return loss
    

    def step(self, env, state):
        action = self.policy(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        self.store_tuple(state, action, reward, new_state, done)
        return action, new_state, reward, done
    

    def initial_fill_buffer(self, env):
        done = True
        while self.buffer.counter < self.batch_size:
            if done:
                state, _ = env.reset()
            action, new_state, reward, done = self.step(env, state)
            state = new_state
        

    def train_model(self, env, num_episodes, steps_per_episode, path):
        os.makedirs(path, exist_ok=False)

        scores, episodes, avg_scores, losses, steps = [], [], [], [], []
        best_avg_score = 150.0  # -np.inf

        # First, fill the buffer following the epsilon-greedy policy
        print(f"INFO: Filling the buffer... (until {self.batch_size} tuples are stored)")
        self.initial_fill_buffer(env)

        log_file = os.path.join(path, "training_log.txt")
        with open(log_file, "a+") as f:
            f.write("Episode,Score,Average_Score,Loss,Steps,Time,Epsilon\n")

            # Now, train the network for num_episodes
            print("INFO: Starting training...")
            for episode in range(1, num_episodes+1):
                done = False
                score = 0.0
                episode_loss = []
                state, _ = env.reset()
                time_elapsed = time.perf_counter()

                for step in range(1, steps_per_episode+1):
                    action = self.policy(state)
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    score += reward
                    self.store_tuple(state, action, reward, new_state, done)
                    state = new_state
                    loss = self.train_step()
                    episode_loss.append(loss)
                    if done:
                        break
                
                time_elapsed = time.perf_counter() - time_elapsed
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_final)
                
                # Store the results
                scores.append(score)
                episodes.append(episode)
                avg_score = np.mean(last_100_scores := scores[-100:])
                best_avg_score = max(avg_score, best_avg_score)
                avg_scores.append(avg_score)
                losses.append(np.mean(episode_loss))
                steps.append(step)

                print(f"Episode {episode: >4}/{num_episodes: <4} -->  Score = {score:8.2f} | AVG Score (last {len(last_100_scores):3}) = {avg_score:8.2f} | "
                      f"Loss = {losses[-1]:5.2f} | Steps = {step:4} | Time = {time_elapsed:5.2f}s | Epsilon = {self.epsilon}")
                
                # Write the metrics to the training log in CSV format
                # with open(log_file, "a+") as f:
                f.write(f"{episode},{score},{avg_score},{losses[-1]},{step},{time_elapsed},{self.epsilon}\n")
                
                # Save the model (and weights) if the average score is the best so far or if it's the last episode
                if avg_score >= best_avg_score or episode == num_episodes:
                    self.q_net.save(backup_path := os.path.join(path, "models", f"dqn_model_{episode}"))
                    self.q_net.save_weights(os.path.join(path, "weights", f"dqn_weights_{episode}", f"dqn_weights_{episode}"))
                    # txt.write("Save {0} - Episode {1}/{2}, Score: {3} ({4}), AVG Score: {5}\n".format(f, i, num_episodes,
                    #                                                                                 score, self.epsilon,
                    #                                                                                 avg_score))
                    # f += 1
                    print(f'INFO: Backing up model to "{backup_path}"')

        print(f"INFO: Training finished")

        df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Threshold': [200]*num_episodes})

        plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
        plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed', label='Media score')
        plt.plot('x', 'Solved Threshold', data=df, marker='', color='red', linewidth=2, linestyle='dashed', label='Umbral resuelto')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(train_graph_path := os.path.join(path, 'TrainMetrics.png'))
        print(f'INFO: Training metrics graph saved to "{train_graph_path}"')


    def test(self, env, num_episodes, path):
        # path_prefix = os.path.split(path)[0]
        print(f"INFO: Loading model at {path}")
        self.q_net = load_model(path)

        self.epsilon = 0.0
        scores, episodes, avg_scores = [], [], []
        score = 0.0

        print("INFO: Starting test...")
        for episode in range(1, num_episodes+1):
            state, _ = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                # env.render()
                action = self.policy(state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_score += reward
                state = new_state
            score += episode_score
            scores.append(episode_score)
            episodes.append(episode)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

        print(f"INFO: Test finished with an average score of {np.mean(avg_scores):.2f} over {num_episodes} episodes")

        df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': [200]*num_episodes})

        plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
        plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed', label='Media score')
        plt.plot('x', 'Solved Threshold', data=df, marker='', color='red', linewidth=2, linestyle='dashed', label='Umbral resuelto')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(test_graph_path := os.path.join(path, 'TestMetrics.png'))
        print(f'INFO: Test metrics graph saved to "{test_graph_path}"')

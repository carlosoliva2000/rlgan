from datetime import datetime
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()

from agent import Agent
import gymnasium as gym
import os


def main():
    env = gym.make("LunarLander-v2")
    num_episodes = 500
    steps_per_episode = env.spec.max_episode_steps  # 800

    dqn_agent = Agent(
        lr=0.00075, 
        discount_factor=0.99, 
        num_actions=4, 
        batch_size=64, 
        input_dims=8, 
        hidden_dims=[32, 32]
    )

    dqn_agent.train_model(
        env=env, 
        num_episodes=num_episodes, 
        steps_per_episode=steps_per_episode,
        path=os.path.abspath(os.path.join(os.getcwd(), 'DQN', 'saved_networks', f'model{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'))
    )

    env.close()


if __name__ == '__main__':
    main()

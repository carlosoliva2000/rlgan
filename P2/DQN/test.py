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
    env = gym.make("LunarLander-v2", render_mode="rgb_array")  # , render_mode="human")
    num_episodes = 10

    dqn_agent = Agent(
        lr=0.00075, 
        discount_factor=0.99, 
        num_actions=4, 
        batch_size=64, 
        input_dims=8, 
        hidden_dims=[32, 32]
    )
    
    dqn_agent.test(
        env=env, 
        num_episodes=num_episodes, 
        path=os.path.abspath(os.path.join(os.getcwd(), 'saved_networks', 'model2024-02-25-14-25-58', 'models', 'dqn_model_500'))  # os.path.abspath(os.path.join(os.getcwd(), 'saved_networks', 'model2024-02-24-19-32-12', 'models', 'dqn_model_500'))
    )

    env.close()


if __name__ == '__main__':
    main()

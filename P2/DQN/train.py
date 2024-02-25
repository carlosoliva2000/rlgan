import os

def main():
    from datetime import datetime
    from agent import Agent

    import gymnasium as gym

    env = gym.make("LunarLander-v2")
    assert env is not None, "Environment not found"
    assert env.spec is not None, "Environment spec not found"
    num_episodes = 500
    steps_per_episode = env.spec.max_episode_steps  # 800

    dqn_agent = Agent(
        lr=0.00075, 
        discount_factor=0.99, 
        num_actions=4, 
        batch_size=64, 
        input_dims=8, 
        epsilon_decay=0.99,
        # epsilon_decay=0.92,
        # update_rate=2000,
        hidden_dims=[64, 64]  # [32, 32]
    )

    dqn_agent.train_model(
        env=env, 
        num_episodes=num_episodes, 
        steps_per_episode=steps_per_episode,
        path=os.path.abspath(os.path.join(os.getcwd(), 'saved_networks', f'model{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'))
    )

    env.close()


def check_cwd():
    script_hint = "DQN"
    msg = f"""ERROR: You need to set your cwd to \"{os.path.dirname(os.path.abspath(__file__))}\" to execute this script.
       This ensures that the path to save the results is correct.
       Then, you can run the train.py script"""

    # Check if the file is there
    if not os.path.isfile("_EXECUTE_FROM_THIS_PATH.txt"):
        print(msg)
        return False
    
    # Check if the line is script_hint
    with open("_EXECUTE_FROM_THIS_PATH.txt", "r") as f:
        line = f.readline().strip()
        if line != script_hint:
            return False
    return True


if __name__ == '__main__':
    if check_cwd():
        main()

from typing import Optional
from src.MLP import MLP
from src.LunarLanderSolver import LunarLanderSolver
from PIL import ImageDraw, ImageFont

import numpy as np
import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt
import PIL.Image
import imageio

def run (model: MLP, video, episode: int, seed=None, norm_reward=False, visualize=False):
    def policy (observation, model):
        s = model.forward(observation)
        action = np.argmax(s)
        return action
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default(14)
    
    env = gym.make("LunarLander-v2", render_mode="human" if visualize else "rgb_array")
    #observation, info = env.reset(seed=42)
    observation, info = env.reset(seed=int(seed) if seed is not None else None)
    video.append_data(env.render())
    steps = 0
    nonzero_actions = 0
    racum = 0.0
    while True:
        action = policy(observation, model)
        # action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        racum += reward # type: ignore
        steps += 1
        if action != 0:
            nonzero_actions += 1
        # print(reward)
            
        render = PIL.Image.fromarray(env.render())
        draw = ImageDraw.Draw(render)
        draw.text((5.0, 5.0), f"Episode: {episode:2}", font=font)
        draw.text((5.0, 20.0), f"Steps: {steps:4}", font=font)
        draw.text((5.0, 35.0), f"Reward: {reward:7.2f}", font=font)
        draw.text((5.0, 50.0), f"Racum: {racum:7.2f}", font=font)
        draw.text((render.width-5.0, 5.0), f"Seed: {seed}", font=font, anchor="rt")
        video.append_data(np.array(render))

        if terminated or truncated:
            draw.text((5.0, 70.0), "FINISHED", font=font)
            render_array = np.array(render)
            for _ in range(90):
                video.append_data(render_array)
            output = {'racum': (racum + 500) / 700 if norm_reward else racum,
                      'steps': steps,
                      'nonzero_actions': nonzero_actions}
            return output
        

def check_cwd():
    script_hint = "NEUROEVOLUTION"
    msg = f"""ERROR: You need to set your cwd to \"{os.path.dirname(os.path.abspath(__file__))}\" to execute this script.
       This ensures that the path to load the results is correct.
       Then, you can run the test.py script"""

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


def test():
    try:
    #     ch = np.array(
    #         # [0.5976300923982026, 0.18851473499679905, -0.8771923475590399, 0.6412757509893395, 0.8968001830470116, 0.6406366555725092, -0.12931230679157357, -1.0, -0.3283377352544007, -0.40227815538538647, -0.9612003612205076, -0.04609752102318576, 0.1483569847509053, -0.3334436517758156, -0.8944409379259455, 0.9873309507241416, -0.1507749713887343, -0.8471137925921733, 1.0, -0.6927305080503819, 0.266953912504062, 1.0, -0.6673182368711582, 0.6656007301984435, -1.0, -1.0, 1.0, -1.0, 0.027184123698695273, 0.541915671604348, -0.3611107918839286, -0.18299766664034112, 1.0, 0.11501556328860452, -0.12274314160942712, -0.05741746604617182, -0.2515464811953385, -0.7220439466413223, -0.8451994106496745, -1.0, 1.0, -0.9800150679717436, -0.4270769550920527, 0.8054044476653982, 0.7596021418830408, 1.0, 0.735689306806003, 0.046105161552748686, 0.5907346803032923, 0.5447565695977035, 0.011290216902042682, -0.6650346868568274, -0.20117366130488523, -0.874550363365975, -0.22799283627771572, 1.0, 0.024873698118908938, 0.22634335589110233, -1.0, 1.0, 0.7707858628566095, -0.4553339992602988, 1.0, -1.0, -0.22857438329043622, 0.07544961813005457, 1.0, 0.6529946254990275, -0.6889907618199412, -0.383759229180033, -1.0, -1.0, 0.9208572504790575, -0.2577680796010897, -0.0038158897517339696, -0.3180968931388999, -0.55768826941003, -0.7050883838300236, 0.2350061983834686, -0.6861258756227548, -0.9777720982062632, -0.5415542466148922]
    # # [0.033481954686337634, 0.18070834405061809, 0.33581698127293336, -0.759704308526605, -0.4748104012222619, 0.4630068265672821, -0.39844513164041473, -0.27485982494714156, -0.7764488017790588, 0.5584292966922935, 0.7841530877741325, 0.6886956615213794, -0.9275918118413531, 0.5133040684855795, -0.2496924311410621, -1.0, -1.0, -1.0, -1.0, 0.5136366289696734, -0.9683106541535634, -0.3066289122948911, 0.5706830797841306, 1.0, 1.0, -0.567600107979674, -0.7366911177444204, 0.9284242567787272, 0.28623917138605104, 0.3009314290311177, 0.9334684468477907, 0.5197655392029173, -1.0, -0.11845814693555179, 1.0, 0.25304386903797815, -0.3974937132752022, 0.5813817265217051, -0.1352488515967879, 0.12324447757262402, 0.5708951192150672, -1.0, 0.22652834588578286, 0.7605830409095065, -0.16545532252899342, 0.737636689275929, 0.3529424190704753, -0.43080175994408587, 0.18066696516052205, -0.33911964269362493, -0.5510044438572951, 0.22745875631047519, 0.6580640224357583, -0.45363860896150043, -0.797056743672436, -0.7282249216881711, 0.9082151844946659, 0.6809186573930167, 1.0, 0.8961257031138085, -0.5368321457152166, 1.0, -0.1431776573871102, -0.011269134386448831, 0.6797686257992449, -0.619328282408667, 0.3007294268691746, 0.6559249542137099, 1.0, 1.0, -0.8263561741485244, 0.08258717002667204, -1.0, -0.10157796972734923, 1.0, 0.8711115301331255, -1.0, 0.9491425269847683, -1.0, 0.3246928706236024, 0.8608012130862249, -0.49685688312892795]
    #     [0.29147150004090217, -0.9435972617054909, 0.3300504133693529, -0.6562731537632738, 0.6911027719967239, 0.35075729876715056, -0.742284998186917, -0.677795325255774, -0.9355360268631796, 0.7208737475018698, -1.0, -0.13851850144599998, 0.39767434516634165, -0.2779962069201042, 0.3469812786741382, 0.12083202481110822, 0.8651691591165431, 0.7603723070581072, 0.47099608359136413, 0.5300271223129336, -0.6982580783437938, -0.4189139531078295, -0.6901660942030847, 1.0, -0.0292666940218686, 0.6348540375081135, 0.16148137361877718, 0.5220802530544717, -0.9168409039398058, -0.5146992736956408, 0.15478737352012423, 0.8597632428146399, -0.6052834991672673, -0.0386777390512148, -0.7622466527255344, 0.17062400345505707, -0.6135532908628196, -0.31331275776851936, -0.16077449825192663, -0.06377237055971274, 0.5592959075413161, -0.6977355020638508, -0.6086929933961633, 0.3942662954441748, -0.36716891629020765, -0.30938095258617276, -0.35686982948258594, 0.9581505090916669, 0.31909967319181726, 0.098707953694183, 0.3844385178706854, -0.6506849008000082, -0.22020830639459138, -0.7688271584645454, -1.0, 0.14601074805198255, -0.0069545129764312374, -0.120030108547384, 0.37644974450290586, -0.58570653337536, -0.37512214858258314, 0.4983454310350662, -0.11163624992026855, 0.8167117806444886, 0.9582757056187958, 0.717285007547459, -0.4956694568961871, 0.20369803201673858, 1.0, -0.45998530452684316, -0.3153579905168601, 0.9515441201400141, 0.9147803738579272, -0.8637880272898524, 0.06934900347090567, 0.8754542399532609, -1.0, -0.06290992820248259, 0.40026496288458613, -1.0, -0.9022208350155125, -0.0723659256249809]
    #     )
        PREFIX_PATH = "results/best_lunarlander_2024-02-05_23-36"  # "results/best_lunarlander_2024-02-24_17-39"  # "results/best_lunarlander_2024-02-06_10-44"
        EPOCHS = 10
        VISUALIZE = False

        ####################################################################################################
        PREFIX_PATH = os.path.abspath(PREFIX_PATH)
        ch = np.load(os.path.join(PREFIX_PATH, "ch.npy"))
        model = MLP(
            layers=[8, 6, 4],
            chromosome=ch
            )
        
        print(f"Model with the chromosome at {PREFIX_PATH} loaded")
        print(f"Starting test for {EPOCHS} epochs...\n")
        
        scores = []
        steps = []
        nonzero_actions = []
        seeds = np.random.randint(0, 2**32, EPOCHS, dtype=np.uint64)

        with imageio.get_writer(os.path.join(PREFIX_PATH, "test_video.mp4"), fps=50) as video:
            for i in range(EPOCHS):
                result = run(model, video, episode=i, seed=seeds[i], visualize=VISUALIZE)
                act_steps_ratio = result['nonzero_actions'] / result['steps']
                print(f"Epoch {i:2} --> Steps: {result['steps']:4} | Non-zero actions: {result['nonzero_actions']:4} | "
                    f"Actions/Steps: {act_steps_ratio:.4f} | Score: {result['racum']}")
                scores.append(result['racum'])
                steps.append(result['steps'])
                nonzero_actions.append(result['nonzero_actions'])
                if VISUALIZE:
                    time.sleep(1)
        
        print("\nTest finished")
        print(f"Average score: {np.mean(scores):.4f}")
        print(f"Median score: {np.median(scores):.4f}")
        print(f"Average steps: {np.mean(steps):.2f}")
        print(f"Average non-zero actions: {np.mean(nonzero_actions):.2f}")
        print(f"Average actions/steps ratio: {(np.mean(nonzero_actions) / np.mean(steps)):.4f}\n") # type: ignore
        
        try:
            lunarlander_solver = LunarLanderSolver()
            lunarlander_solver.avg_fitnesses = np.load(os.path.join(PREFIX_PATH, "avg_fitnesses.npy"))
            lunarlander_solver.max_fitnesses = np.load(os.path.join(PREFIX_PATH, "max_fitnesses.npy"))
            lunarlander_solver.min_fitnesses = np.load(os.path.join(PREFIX_PATH, "min_fitnesses.npy"))
            lunarlander_solver.times_elapsed = np.load(os.path.join(PREFIX_PATH, "times_elapsed.npy"))
            lunarlander_solver.plot_fitness(show_min=False, show_avg=False)
        except OSError as exc:
            print(f"\nSkipping plotting historic training data due to the following error: {exc}")

        # plot in a figure with 2 subplots, one representings steps vs iteration and the other non-zero actions vs iteration
        print("\nPlotting steps and non-zero actions vs iteration...")
        fig, axs = plt.subplots(2)
        axs[0].plot(nonzero_actions, label="Non-zero actions", marker="o", linestyle="--")
        axs[0].plot(steps, label="Steps", marker="o", linestyle="--")
        axs[0].set_title("Steps and Non-zero actions throughout iterations")
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Steps/Non-zero actions")
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[0].set_yticks(np.arange(0, 1001, 200))
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(np.array(nonzero_actions) / np.array(steps), marker="o", linestyle="--")
        axs[1].set_title("Non-zero actions/Steps ratio")
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Ratio")
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # force the y-axis to be between 0 and 1 with a step of 0.1
        axs[1].set_yticks(np.arange(0, 1.1, 0.1))
        axs[1].grid()
        # axs[1].legend()
        plt.show()
            
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nTest failed due to the following error: {exc}")
    

if __name__ == "__main__":
    if check_cwd():
        test()

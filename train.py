import time
from tracemalloc import start
import numpy as np
from src.MLP import MLP
from src.LunarLanderSolver import LunarLanderSolver


def train():
    print("Initializing training...")
    lunarlander_solver = LunarLanderSolver(norm_reward=True, iter_for_ind=10)
    print("Creating population...")
    pop = lunarlander_solver.create_population_mlp(n=100)
    print("Population created")
    best = lunarlander_solver.evolve(pop, lunarlander_solver.fitness, pmut=0.25, ngen=200,  # pmut=1/82
                            trace=1, T=10, pcross=0.7, beta=0.4, elitism=False)
    fitness = lunarlander_solver.fitness(best) if best is not None else np.nan

    if best is not None:
        best_ch = best.to_chromosome()
        print(f"\n\nFitness: {fitness}")
        print(f"Best chromosome: {best_ch.tolist()}")
        np.savetxt("best_chromosome.txt", best_ch)
        np.save("best_chromosome.npy", best_ch)

    # solutions.append(best)
    # fits.append(lunarlander_solver.fitness(best) if best is not None else np.nan)
        
import gymnasium as gym

  # , render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")

def run (model, env):
    # if env is None:
    #     env = gym.make("LunarLander-v2", render_mode="human")
    # else:
    #     print(True)
    def policy (observation, model):
        s = model.forward(observation)
        action = np.argmax(s)
        return action
    #observation, info = env.reset(seed=42)
    observation, info = env.reset()
    ite = 0
    racum = 0.0
    while True:
        action = policy(observation, model)
        # action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        racum += reward
        # print(reward)

        if terminated or truncated:
            # r = (racum+500) / 700
            # print(racum, racum+500, r)
            # print(racum, r)
            return (racum+500) / 700

def test_processing():
    model = MLP([8, 6, 4])
    model.from_chromosome([-0.2614334364755576, -0.43715576429143344, 0.3763431912417557, -0.08876552711706162, -0.2342455299285781, -0.05730437568962213, -0.11242302006395466, -0.18721951036209494, 0.2751434097626164, -0.03245785521665117, -0.6504633583397568, -0.4422648718530036, 0.20598057521341112, 0.011667253873116911, 0.537035106243016, 0.3867893099461863, -0.3190719858614685, -0.49309962270270913, -0.18651301399230907, 0.29461509311909484, 0.3552058035148239, 0.07322537261746653, 0.04039636199444518, 0.2603378654994551, 0.2612305563163384, -0.5453466053424453, 0.3540265174667878, -0.6144137087017891, -0.2410840636233743, -0.1975928408264459, -0.22199124570894418, 0.6165850743733697, 0.06758043450249174, -0.3309301997356706, 0.47219647500083617, 0.3125531313205563, -0.3101393385670267, 0.10090486265603094, 0.40821775009882, -0.038901672454861445, 0.32500365014916527, -0.5543128855973511, -0.46663451287612356, -0.20963062287466144, 0.29799040053606335, 0.2363383813995609, 0.2797063130452821, -0.18447101903193017, 0.0524897074185768, -0.03622447606216642, -0.2795239373686361, 0.08248011531546741, -0.28234416776121257, -0.05773220827720571, -0.18909687255207974, -0.6086042380217522, 0.3805064593736499, -0.1678750687366219, 0.7009531880098665, -0.2520856224381874, -0.1807846476046446, -0.324240464563758, -0.29231842214804593, 0.28961644144351606, -0.37731072404748334, 0.11620629251157297, 0.0729098464784065, 0.13457158422521145, 0.24087703983999723, -0.4024780285921873, 0.03913216928711482, -0.3956581932826715, -0.46064090764644317, 0.05348423718291635, -0.36710592579552154, 0.07921140666740453, 0.20355451138921848, 0.17988057398317414, 0.5435600348111508, -0.0826487049234676, -0.5825055589379566, -0.43617699992149406])
    import multiprocessing
    with multiprocessing.Pool(processes=10) as pool:
        # print(pool)
        results = pool.starmap(run, [(model, gym.make("LunarLander-v2")) for _ in range(10)])
    print(results)
    import numpy as np
    print(np.mean(results), np.median(results), np.std(results), np.max(results), np.min(results))
    # for e in results.get():
    #     print(e.get())

from functools import wraps
import time


# def timeit(func):
#     @wraps(func)
#     def timeit_wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         total_time = end_time - start_time
#         print(f'Function {func.__name__} Took {total_time:.4f} seconds')
#         return result
#     return timeit_wrapper

# @timeit
def test_processing_multi_seq():
    model = MLP([8, 6, 4])
    model.from_chromosome([-0.2614334364755576, -0.43715576429143344, 0.3763431912417557, -0.08876552711706162, -0.2342455299285781, -0.05730437568962213, -0.11242302006395466, -0.18721951036209494, 0.2751434097626164, -0.03245785521665117, -0.6504633583397568, -0.4422648718530036, 0.20598057521341112, 0.011667253873116911, 0.537035106243016, 0.3867893099461863, -0.3190719858614685, -0.49309962270270913, -0.18651301399230907, 0.29461509311909484, 0.3552058035148239, 0.07322537261746653, 0.04039636199444518, 0.2603378654994551, 0.2612305563163384, -0.5453466053424453, 0.3540265174667878, -0.6144137087017891, -0.2410840636233743, -0.1975928408264459, -0.22199124570894418, 0.6165850743733697, 0.06758043450249174, -0.3309301997356706, 0.47219647500083617, 0.3125531313205563, -0.3101393385670267, 0.10090486265603094, 0.40821775009882, -0.038901672454861445, 0.32500365014916527, -0.5543128855973511, -0.46663451287612356, -0.20963062287466144, 0.29799040053606335, 0.2363383813995609, 0.2797063130452821, -0.18447101903193017, 0.0524897074185768, -0.03622447606216642, -0.2795239373686361, 0.08248011531546741, -0.28234416776121257, -0.05773220827720571, -0.18909687255207974, -0.6086042380217522, 0.3805064593736499, -0.1678750687366219, 0.7009531880098665, -0.2520856224381874, -0.1807846476046446, -0.324240464563758, -0.29231842214804593, 0.28961644144351606, -0.37731072404748334, 0.11620629251157297, 0.0729098464784065, 0.13457158422521145, 0.24087703983999723, -0.4024780285921873, 0.03913216928711482, -0.3956581932826715, -0.46064090764644317, 0.05348423718291635, -0.36710592579552154, 0.07921140666740453, 0.20355451138921848, 0.17988057398317414, 0.5435600348111508, -0.0826487049234676, -0.5825055589379566, -0.43617699992149406])
    envs = [gym.make("LunarLander-v2") for _ in range(10)]
    # print([(model, env) for env in envs][0])
    # return
    global_results = []
    start_time = time.perf_counter()
    for i in range(100):
        for e in envs:
            result = run(model, e)
            global_results.append(result)
    end_time = time.perf_counter()
    print(f'Function test_processing_multi_seq Took {end_time - start_time:.4f} seconds')
    results = global_results
    return end_time - start_time
    # print(results)
    # import numpy as np
    # print(np.mean(results), np.median(results), np.std(results), np.max(results), np.min(results))

# @timeit
def test_processing_multi():
    model = MLP([8, 6, 4])
    model.from_chromosome([-0.2614334364755576, -0.43715576429143344, 0.3763431912417557, -0.08876552711706162, -0.2342455299285781, -0.05730437568962213, -0.11242302006395466, -0.18721951036209494, 0.2751434097626164, -0.03245785521665117, -0.6504633583397568, -0.4422648718530036, 0.20598057521341112, 0.011667253873116911, 0.537035106243016, 0.3867893099461863, -0.3190719858614685, -0.49309962270270913, -0.18651301399230907, 0.29461509311909484, 0.3552058035148239, 0.07322537261746653, 0.04039636199444518, 0.2603378654994551, 0.2612305563163384, -0.5453466053424453, 0.3540265174667878, -0.6144137087017891, -0.2410840636233743, -0.1975928408264459, -0.22199124570894418, 0.6165850743733697, 0.06758043450249174, -0.3309301997356706, 0.47219647500083617, 0.3125531313205563, -0.3101393385670267, 0.10090486265603094, 0.40821775009882, -0.038901672454861445, 0.32500365014916527, -0.5543128855973511, -0.46663451287612356, -0.20963062287466144, 0.29799040053606335, 0.2363383813995609, 0.2797063130452821, -0.18447101903193017, 0.0524897074185768, -0.03622447606216642, -0.2795239373686361, 0.08248011531546741, -0.28234416776121257, -0.05773220827720571, -0.18909687255207974, -0.6086042380217522, 0.3805064593736499, -0.1678750687366219, 0.7009531880098665, -0.2520856224381874, -0.1807846476046446, -0.324240464563758, -0.29231842214804593, 0.28961644144351606, -0.37731072404748334, 0.11620629251157297, 0.0729098464784065, 0.13457158422521145, 0.24087703983999723, -0.4024780285921873, 0.03913216928711482, -0.3956581932826715, -0.46064090764644317, 0.05348423718291635, -0.36710592579552154, 0.07921140666740453, 0.20355451138921848, 0.17988057398317414, 0.5435600348111508, -0.0826487049234676, -0.5825055589379566, -0.43617699992149406])
    envs = [gym.make("LunarLander-v2") for _ in range(10)]
    # print([(model, env) for env in envs][0])
    # return
    import multiprocessing
    global_results = []
    start_time = time.perf_counter()
    for i in range(100):
        with multiprocessing.Pool(processes=1) as pool:
            # print(pool)
            # results = [0.0, 1.0, 2.0]
            results = pool.starmap(run, [(model, env) for env in envs])
            # print(results)
            global_results.extend(results)
    end_time = time.perf_counter()
    print(f'Function test_processing_multi Took {end_time - start_time:.4f} seconds')
    results = global_results
    # print(results)
    # import numpy as np
    # print(np.mean(results), np.median(results), np.std(results), np.max(results), np.min(results))

# @timeit
def test_processing_multi_improved():
    model = MLP([8, 6, 4])
    model.from_chromosome([-0.2614334364755576, -0.43715576429143344, 0.3763431912417557, -0.08876552711706162, -0.2342455299285781, -0.05730437568962213, -0.11242302006395466, -0.18721951036209494, 0.2751434097626164, -0.03245785521665117, -0.6504633583397568, -0.4422648718530036, 0.20598057521341112, 0.011667253873116911, 0.537035106243016, 0.3867893099461863, -0.3190719858614685, -0.49309962270270913, -0.18651301399230907, 0.29461509311909484, 0.3552058035148239, 0.07322537261746653, 0.04039636199444518, 0.2603378654994551, 0.2612305563163384, -0.5453466053424453, 0.3540265174667878, -0.6144137087017891, -0.2410840636233743, -0.1975928408264459, -0.22199124570894418, 0.6165850743733697, 0.06758043450249174, -0.3309301997356706, 0.47219647500083617, 0.3125531313205563, -0.3101393385670267, 0.10090486265603094, 0.40821775009882, -0.038901672454861445, 0.32500365014916527, -0.5543128855973511, -0.46663451287612356, -0.20963062287466144, 0.29799040053606335, 0.2363383813995609, 0.2797063130452821, -0.18447101903193017, 0.0524897074185768, -0.03622447606216642, -0.2795239373686361, 0.08248011531546741, -0.28234416776121257, -0.05773220827720571, -0.18909687255207974, -0.6086042380217522, 0.3805064593736499, -0.1678750687366219, 0.7009531880098665, -0.2520856224381874, -0.1807846476046446, -0.324240464563758, -0.29231842214804593, 0.28961644144351606, -0.37731072404748334, 0.11620629251157297, 0.0729098464784065, 0.13457158422521145, 0.24087703983999723, -0.4024780285921873, 0.03913216928711482, -0.3956581932826715, -0.46064090764644317, 0.05348423718291635, -0.36710592579552154, 0.07921140666740453, 0.20355451138921848, 0.17988057398317414, 0.5435600348111508, -0.0826487049234676, -0.5825055589379566, -0.43617699992149406])
    envs = [gym.make("LunarLander-v2") for _ in range(10)]
    # print([(model, env) for env in envs][0])
    # return
    global_results = []
    import multiprocessing
    with multiprocessing.Pool(processes=10) as pool:
        start_time = time.perf_counter()
        for i in range(100):
            # print(pool)
            # results = [0.0, 1.0, 2.0]
            results = pool.starmap(run, [(model, env) for env in envs])
            # print(results)
            global_results.extend(results)
        end_time = time.perf_counter()
        print(f'Function test_processing_multi_improved Took {end_time - start_time:.4f} seconds')
        results = global_results
    return end_time - start_time
    # print(results)
    # import numpy as np
    # print(np.mean(results), np.median(results), np.std(results), np.max(results), np.min(results))

def increment(x):
    return x+1

def run_increment():
    import multiprocessing
    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(increment, range(10))
    print(results)

if __name__ == "__main__":
    train()
    # seq_times = []
    # multi_times = []
    # for _ in range(10):
    #     seq_times.append(test_processing_multi_seq())
    #     multi_times.append(test_processing_multi_improved())
    # print("Mean of seq times:", np.mean(seq_times))
    # print("Mean of multi times:", np.mean(multi_times))
    # test_processing_multi_seq()
    # test_processing_multi_improved()
    # test_processing_multi()
    # test_processing()
    # train()

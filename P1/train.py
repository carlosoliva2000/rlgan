from src.LunarLanderSolver import LunarLanderSolver

import numpy as np
import os


def train():
    print("Initializing training...")
    lunarlander_solver = LunarLanderSolver(norm_reward=False, iter_for_ind=10)
    print("Creating population...")
    pop = lunarlander_solver.create_population_mlp(n=100)
    print("Population created")
    best = lunarlander_solver.evolve(pop, lunarlander_solver.fitness, pmut=0.25, ngen=200,  # pmut=1/82
                                     T=40, pcross=0.7, mutate_factor=0.25, trace=1)  # , early_stop=200.0)
    fitness = lunarlander_solver.fitness(best) if best is not None else np.nan

    if best is not None:
        best_ch = best.to_chromosome()
        print(f"\n\nFitness: {fitness}")
        print(f"Best chromosome: {best_ch.tolist()}")

        lunarlander_solver.plot_fitness()
        input("\nPress Enter to continue...")


def check_cwd():
    script_hint = "NEUROEVOLUTION"
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


if __name__ == "__main__":
    if check_cwd():
        train()

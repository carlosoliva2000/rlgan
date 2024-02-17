import numpy as np
from src.LunarLanderSolver import LunarLanderSolver


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


if __name__ == "__main__":
    train()

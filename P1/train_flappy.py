import numpy as np
from src.FlappyBirdSolver import FlappyBirdSolver


def train():
    print("Initializing training...")
    flappybird_solver = FlappyBirdSolver(norm_reward=False, iter_for_ind=5)  # 10
    print("Creating population...")
    pop = flappybird_solver.create_population_mlp(n=100)
    print("Population created")
    # best = flappybird_solver.evolve(pop, flappybird_solver.fitness, pmut=0.25, ngen=2000,  # pmut=1/82
    #                                  T=70, pcross=0.7, mutate_factor=2.00, trace=1, elitism=False)  # , early_stop=200.0)
    best = flappybird_solver.evolve(pop, flappybird_solver.fitness, pmut=0.7, ngen=2000,  # pmut=1/82
                                     T=40, pcross=0.7, mutate_factor=0.1, trace=1, elitism=True, early_stop=30.0)  # , early_stop=200.0)
    fitness = flappybird_solver.fitness(best) if best is not None else np.nan

    if best is not None:
        best_ch = best.to_chromosome()
        print(f"\n\nFitness: {fitness}")
        print(f"Best chromosome: {best_ch.tolist()}")

        flappybird_solver.plot_fitness()
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    train()

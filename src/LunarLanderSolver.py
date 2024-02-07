import copy
import random
import time
from typing import Callable, Optional
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
import multiprocessing
import os

from .MLP import MLP


class LunarLanderSolver:

    def __init__(self, iter_for_ind=10, norm_reward=False) -> None:
        self.best_ind = None
        self.best_fit = -np.inf
        self.iters_for_ind = iter_for_ind
        self.envs = [gym.make("LunarLander-v2") for _ in range(self.iters_for_ind)]
        self.norm_reward = norm_reward
        self.pool = multiprocessing.Pool(processes=10)
        self.max_fitnesses = []
        self.min_fitnesses = []
        self.avg_fitnesses = []
        self.times_elapsed = []

    def __getstate__(self) -> object:
        # Needed to pickle the object! (because of the multiprocessing.Pool)
        state = self.__dict__.copy()
        del state["pool"]
        return state
    
    @property
    def generations(self):
        return len(self.max_fitnesses)
    
    def plot_fitness(self, show_max=True, show_avg=False, show_min=False):
        if not (show_max or show_avg or show_min):
            return
        
        plt.ion()
        plt.cla()
        if show_max:
            plt.plot(self.max_fitnesses, label="Max Fitness", marker=".")
        if show_avg:
            plt.plot(self.avg_fitnesses, label="Avg Fitness", linestyle="dashed")
        if show_min:
            plt.plot(self.min_fitnesses, label="Min Fitness", marker=".")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(f"Fitness over generations (Generation: {self.generations})")
        plt.legend()
        plt.show()

    def policy(self, model: MLP, observation):
        s = model.forward(observation)
        action = np.argmax(s)
        return action

    def run(self, model: MLP, env: gym.Env):
        observation, info = env.reset()
        ite = 0
        racum = 0.0
        while True:
            action = self.policy(model, observation)
            # action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            racum += reward # type: ignore
            # print(reward)

            if terminated or truncated:
                # r = (racum+500) / 700
                # print(racum, racum+500, r)
                # print(racum, r)
                return racum if not self.norm_reward else (racum+500) / 700
    
    def create_population_mlp(self, n=100, layers: Optional[list[int]]=None) -> list[MLP]:
        """Creates a population of `n` individuals with a default chromosome of 8 input neurons, 
        6 hidden neurons and 4 output neurons (`layers=[8, 6, 4]`).

        Args:
            n (int, optional): _description_. Defaults to 100.
            layers (Optional[list[int]], optional): _description_. Defaults to [8, 6, 4] if None.

        Returns:
            list[MLP]: a list of `n` individuals of type `MLP`.
        """
        if layers is None:
            layers = [8, 6, 4]
        pop = [MLP(layers) for _ in range(n)]
        return pop
    
    def sort_pop(self, pop: list[MLP], fit: Callable[..., float]) -> tuple[list[float], list[MLP]]:
        """Sorts a population `pop` by the fitness function `fit` and returns the fitness of each individual and the sorted
        population.

        Args:
            pop (list[MLP]): _description_
            fit (Callable[..., float]): _description_

        Returns:
            tuple[list[float], list[MLP]]: two parallel lists with the fitness of each individual and the sorted population (from best to worst fitness)
        """
        pop_sorted = sorted(pop, key=fit, reverse=True)
        fitness = [x.mean_fitness for x in pop_sorted]
        return fitness, pop_sorted

    def crossover(self, ind1: MLP, ind2: MLP, pcross: float) -> tuple[MLP, MLP]:
        """Crosses two individuals with a probability `pcross` and returns two new individuals. The new individuals are
        created by taking the first `split_point` genes from `ind1` and the rest from `ind2` and vice versa.

        Args:
            ind1 (MLP): _description_
            ind2 (MLP): _description_
            pcross (float): _description_

        Returns:
            tuple[MLP, MLP]: the new individuals (either `ind1` and `ind2` or two new individuals with the genes of both parents)
        """
        if random.random() < pcross:
            ind1_ch = ind1.to_chromosome().tolist()
            ind2_ch = ind2.to_chromosome().tolist()

            # create a split point for the crossover
            split_point = random.randint(0, len(ind1_ch) - 1)
            new_ind1 = ind1_ch[:split_point] + ind2_ch[split_point:]
            new_ind2 = ind2_ch[:split_point] + ind1_ch[split_point:]

            return MLP(layers=[8, 6, 4], chromosome=new_ind1), MLP(layers=[8, 6, 4], chromosome=new_ind2)
        return ind1, ind2

    def mutate(self, ind: MLP, pmut: float, mutate_factor=0.25, range_values=(-1.0, 1.0)):
        """Mutates an individual with a probability `pmut`, adding a `mutate_factor` multiplied by a random number
        following a uniform distribution between -1 and 1 to each gene of the individual's chromosome. The result is
        clipped to the `range_values`.

        Args:
            ind (MLP): individual to mutate
            pmut (float): probability of mutation for each gene
            mutate_factor (float, optional): factor. Defaults to 0.25.
            rang (tuple, optional): _description_. Defaults to (-1.0, 1.0).

        Returns:
            _type_: _description_
        """
        ind_ch = ind.to_chromosome()
        # create a random number between -1 and 1 with numpy
        for i in range(len(ind_ch)):
            if random.random() < pmut:
                # add a random number to ind[i] with a normal distribution with mean=0 and std=1 and make sure it is in the range
                ind_ch[i] = np.clip(ind_ch[i] + np.random.uniform(-1, 1) * mutate_factor, range_values[0], range_values[1])
        return ind.from_chromosome(ind_ch)
    
    def fitness(self, model: MLP) -> float:
        """Calculates the fitness of an individual `model` by running it in the LunarLander environment (calling `self.run`) and returning the
        mean fitness of `iters_for_ind` iterations. The mean fitness is stored in the individual `model.mean_fitness` and returned.

        The fitness is only calculated if the individual has not been evaluated before. This is done to avoid a bad sorting of the
        population when calling `self.sort_pop` due to inconsistencies in the fitness of the individuals.

        To optimize the evaluation of the fitness of the individual, the method uses a multiprocessing pool to run the
        individual in parallel in the different environments.

        Args:
            model (MLP): _description_

        Returns:
            float: the mean fitness of the individual.
        """
        if model.mean_fitness != -np.inf:
            return model.mean_fitness
        results = self.pool.starmap(self.run, [(model, env) for env in self.envs])
        model.mean_fitness = float(np.mean(results))
        return model.mean_fitness
        # r = 0.0
        # for i in range(self.iters_for_ind):
        #     r += self.run(model, self.envs[i])
        # return r / self.iters_for_ind


    def select_tournament(self, pop: list[MLP], t: int) -> MLP:
        """Selects the best individual from a random sample of `t` individuals from a *sorted* population `pop` using a tournament
        selection. The best individual is the one with the highest fitness.

        Args:
            pop (list[MLP]): _description_
            t (int): _description_

        Returns:
            MLP: the best individual from the tournament.
        """
        sample = random.sample(pop, t)
        sample.sort(key=pop.index)
        return sample[0]  # copy.deepcopy(sample[0])
        # return max(sample, key=lambda x: fitness(x, onlyone))

    def evolve_iteration(self, 
                         pop: list[MLP], 
                         fit: Callable[..., float], 
                         pmut: float, pcross: float=0.7, 
                         T: int=2, mutate_factor: float=0.25, elitism: bool=False, do_print: bool=False) -> tuple[list[float], list[MLP]]:
        """Evolves a population `pop` for one generation using a tournament selection, one-point crossover and random-addition mutation.
        The new population is sorted by fitness and returned.

        Args:
            pop (list[MLP]): _description_
            fit (Callable[..., float]): _description_
            pmut (float): _description_
            pcross (float, optional): _description_. Defaults to 0.7.
            T (int, optional): _description_. Defaults to 2.
            mutate_factor (float, optional): _description_. Defaults to 0.25.
            elitism (bool, optional): _description_. Defaults to False.
            do_print (bool, optional): _description_. Defaults to False.

        Returns:
            tuple[list[float], list[MLP]]: the fitness of the new population and the new population sorted by fitness.
        """
        new_pop = [] if not elitism else [pop[0]]  # Si hay elitismo se añade inicialmente el mejor de la población anterior
        len_pop = len(pop)
        if do_print:
            print("crossover, mutation, ", end="")
        while len(new_pop) < len_pop:  # Se generan tantos hijos como padres haya
            parent1 = self.select_tournament(pop, T)
            parent2 = self.select_tournament(pop, T)

            child1, child2 = self.crossover(parent1, parent2, pcross)
            child1.reset_fitness()
            child2.reset_fitness()

            child1 = self.mutate(child1, pmut, mutate_factor)
            child2 = self.mutate(child2, pmut, mutate_factor)
            
            new_pop.append(child1)
            if len(new_pop) < len_pop:
                new_pop.append(child2)
        
        print("final sorting...", end="")
        fitness, pop = self.sort_pop(new_pop, fit)
        return fitness, pop

    def evolve(self,
               pop: list[MLP], 
               fit: Callable[..., float],
               ngen=100, T=10,
               pcross: float=0.7, pmut: float=0.25,
               mutate_factor=0.25, elitism=False,
               plt_show=False, save_best=True, save_best_path=f"results/best_lunarlander_{time.strftime('%Y-%m-%d_%H-%M')}",
               early_stop: Optional[float]=None,
               trace: int=0
               ) -> MLP | None:
        """Evolves a population `pop` during `ngen` generations to solve the LunarLander problem and returns the best `MLP` found.

        Args:
            pop (list[MLP]): _description_
            fit (Callable[..., float]): _description_
            ngen (int, optional): _description_. Defaults to 100.
            T (int, optional): _description_. Defaults to 10.
            pcross (float, optional): _description_. Defaults to 0.7.
            pmut (float, optional): _description_. Defaults to 0.25.
            mutate_factor (float, optional): _description_. Defaults to 0.25.
            elitism (bool, optional): _description_. Defaults to False.
            plt_show (bool, optional): _description_. Defaults to False.
            save_best (bool, optional): _description_. Defaults to True.
            save_best_path (_type_, optional): _description_. Defaults to f"results/best_lunarlander_{time.strftime('%Y-%m-%d_%H-%M')}".
            early_stop (Optional[float], optional): _description_. Defaults to None.
            trace (int, optional): _description_. Defaults to 0.

        Returns:
            MLP | None: the best individual found.
        """
        try:
            if save_best:
                os.makedirs(save_best_path, exist_ok=False)
            save_best_path = os.path.abspath(save_best_path)

            pop = copy.deepcopy(pop)
            print("Initial population sorting...")
            fitness, pop = self.sort_pop(pop, fit)
            # print(fitness)
            if trace:
                print(f"Generation    0:  (initial)  Best Fitness = {fitness[0]}")

            for generation in range(1, ngen+1):
                do_print = bool(trace and generation % trace == 0)
                if do_print:
                    print(f"Generation {generation:4}:  (", end="")
                start_time = time.perf_counter()
                fitness, pop = self.evolve_iteration(pop, fit, pmut, pcross, T, mutate_factor, elitism, do_print)
                end_time = time.perf_counter()

                # fitness, pop = self.sort_pop(pop, fit)

                time_elapsed = end_time - start_time

                self.times_elapsed.append(time_elapsed)
                self.max_fitnesses.append(fitness[0])
                self.min_fitnesses.append(fitness[-1])
                self.avg_fitnesses.append(np.mean(fitness))

                # Record statistics
                if fitness[0] > self.best_fit:
                    self.best_ind = pop[0]
                    self.best_fit = fitness[0]
                    if save_best:
                        best_ch = self.best_ind.to_chromosome()
                        np.save(os.path.join(save_best_path, "ch.npy"), best_ch)
                        np.savetxt(os.path.join(save_best_path, "ch.txt"), best_ch)
                    if early_stop and fitness[0] > early_stop:
                        print(f")  Best Fitness = {fitness[0]:0.16f}  (time elapsed: {time_elapsed:4.4f}s) (early stop due to fitness > {early_stop})")
                        raise KeyboardInterrupt

                if do_print:
                    print(f")  Best Fitness = {fitness[0]:0.16f}  (time elapsed: {time_elapsed:4.4f}s)")
                    # print(fitness)
                    # print()
                    if plt_show:
                        plt.ion()
                        plt.cla()
                        plt.plot(self.max_fitnesses, label="Max Fitness")
                        plt.plot(self.avg_fitnesses, label="Avg Fitness")
                        plt.plot(self.min_fitnesses, label="Min Fitness")
                        plt.xlabel("Generation")
                        plt.ylabel("Fitness")
                        plt.title(f"Fitness over generations (Generation: {generation})")
                        plt.legend()
                        # plt.show(block=False)
                        plt.pause(0.01)

        except KeyboardInterrupt:
            return self.best_ind
        finally:
            np.save(os.path.join(save_best_path, "max_fitnesses.npy"), self.max_fitnesses)
            np.save(os.path.join(save_best_path, "min_fitnesses.npy"), self.min_fitnesses)
            np.save(os.path.join(save_best_path, "avg_fitnesses.npy"), self.avg_fitnesses)
            np.save(os.path.join(save_best_path, "times_elapsed.npy"), self.times_elapsed)

        return pop[0]  # a.k.a. best

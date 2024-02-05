import copy
import random
import time
from typing import Callable, Optional, Union
import numpy as np
import gymnasium as gym
import multiprocessing

from .MLP import MLP
# from copy import deepcopy

# pool = multiprocessing.Pool(processes=10)


class LunarLanderSolver:

    def __init__(self, iter_for_ind=10, norm_reward=False) -> None:
        self.best = None
        self.iters_for_ind = iter_for_ind
        self.envs = [gym.make("LunarLander-v2") for _ in range(self.iters_for_ind)]
        self.norm_reward = norm_reward
        self.pool = multiprocessing.Pool(processes=10)

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["pool"]
        return state
    
    def set_render_mode(self, mode: str):
        self.env.close()
        self.env = gym.make("LunarLander-v2", render_mode=mode)

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
    
    def create_population_mlp(self, n=100, layers: Optional[list[int]]=None):
        if layers is None:
            layers = [8, 6, 4]
        pop = [MLP(layers) for _ in range(n)]
        return pop
    
    def sort_pop (self, pop: list[MLP], fit: Callable[..., float]): # devuelve una tupla: la lista de fitness y la población ordenada por fitness
        pop_sorted = sorted(pop, key=fit, reverse=True)
        fitness = [fit(x) for x in pop_sorted]
        return fitness, pop_sorted

    def crossover(self, ind1: MLP, ind2: MLP, pcross: float, beta: float):  # devuelve el cruce (emparejamiento) de dos individuos, utilizando BLX (blend crossover)
        if random.random() < pcross:
            ind1_ch = ind1.to_chromosome()
            ind2_ch = ind2.to_chromosome()

            new_ind1 = beta * ind1_ch + (1 - beta) * ind2_ch
            new_ind2 = beta * ind2_ch + (1 - beta) * ind1_ch

            return MLP(layers=[8, 6, 4], chromosome=new_ind1), MLP(layers=[8, 6, 4], chromosome=new_ind2)
        return ind1, ind2

    def mutate(self, ind: MLP, pmut: float, rang=(-1.0, 1.0)):  # devuelve individuo mutado; la mutación consistirá en cambiar un gen en el rango rang (-5, 5)
        ind_ch = ind.to_chromosome()
        # create a random number between -1 and 1 with numpy
        for i in range(len(ind_ch)):
            if random.random() < pmut:
                # add a random number to ind[i] with a normal distribution with mean=0 and std=1 and make sure it is in the range
                ind_ch[i] = np.clip(ind_ch[i] + np.random.uniform(-1, 1) * 0.1, rang[0], rang[1])
        return ind.from_chromosome(ind_ch)
        # decide how many genes to mutate
        # n_gens_to_mutate = int(pmut * len(ind_ch))
        # # select (index) gens to mutate
        # i_gens_to_mutate = random.sample(range(len(ind_ch)), n_gens_to_mutate)
        # # mutate gens
        # for i in i_gens_to_mutate:
        #     ind_ch[i] = random.uniform(*rang)
        # return ind.from_chromosome(ind_ch)
    

        # n_gens_to_mutate = int(pmut * len(ind))
        # # select gens to mutate
        # gens_to_mutate = random.sample(ind, n_gens_to_mutate)
        # # mutate gens
        # for i in range(len(gens_to_mutate)):
        #     ind_to_mutate[i] = random.uniform(*rang)
        # return ind
        # for i in range(len(ind)):
        #     if random.random() < pmut:
        #         # add a random number to ind[i] with a normal distribution with mean=0 and std=1 and make sure it is in the range
        #         ind[i] = np.clip(ind[i] + np.random.normal(0, 1), rang[0], rang[1])
        # return ind

    # def fitness_himmel(self, ch):
    #     return 1 / (1 + self.himmelblau(ch))
    #     # return - self.himmelblau(ch)
    
    def fitness(self, model: MLP) -> float:
        results = self.pool.starmap(self.run, [(model, env) for env in self.envs])
        return np.mean(results)
    
        # r = 0.0
        # for i in range(self.iters_for_ind):
        #     r += self.run(model, self.envs[i])
        # return r / self.iters_for_ind

    # def replace_bottom_population(self, sorted_population, prob_replace):
    #     if random.random() < prob_replace:
    #         # replace bottom 10% of population with new population
    #         new_pop = self.create_population(n=int(len(sorted_population) * 0.1))
    #         sorted_population = sorted_population[:int(len(sorted_population) * 0.9)] + new_pop
    #     return sorted_population

    def select_tournament(self, pop: list[MLP], t: int):
        sample = random.sample(pop, t)
        sample.sort(key=pop.index)
        return copy.deepcopy(sample[0])
        # return max(sample, key=lambda x: fitness(x, onlyone))

    # def search_nearby(self, ind, fit, range=(-5, 5)):
    #     """
    #     Busca en el entorno de un individuo
    #     """
    #     import numpy as np
    #     ind = copy.deepcopy(ind)
    #     best_ind = copy.deepcopy(ind)
    #     for x in np.linspace(-0.5, 0.5, 10).tolist():
    #         for y in np.linspace(-0.5, 0.5, 10).tolist():
    #             new_ind = [
    #                 np.clip(ind[0] + x, range[0], range[1]), 
    #                 np.clip(ind[1] + y, range[0], range[1])
    #                 ]
    #             if fit(new_ind) > fit(best_ind):
    #                 best_ind = new_ind
    #     return best_ind

    def evolve_iteration(self, 
                         pop: list[MLP], 
                         fit: Callable[..., float], 
                         pmut: float, pcross: float=0.7, 
                         T: int=2, mutate_times: int=1, beta: float=0.4, elitism: bool=False, do_print: bool=False):
        """
        Evoluciona una población una generación
        """
        # if do_print:
        #     print("sorting, ", end="")
        # _, pop = self.sort_pop(pop, fit)

        # if random.random() < replace_bottom:
        #     pop = self.replace_bottom_population(pop, replace_bottom)
        
        new_pop = [] if not elitism else [pop[0]]  # Si hay elitismo se añade inicialmente el mejor de la población anterior
        len_pop = len(pop)
        if do_print:
            print("crossover, mutation, ", end="")
        while len(new_pop) < len_pop:  # Se generan tantos hijos como padres haya
            parent1 = self.select_tournament(pop, T)
            parent2 = self.select_tournament(pop, T)

            child1, child2 = self.crossover(parent1, parent2, pcross, beta)
            
            for _ in range(mutate_times):
                child1 = self.mutate(child1, pmut)
                child2 = self.mutate(child2, pmut)
                # if self.himmelblau(new_child1) < self.himmelblau(child1):
                #     child1 = new_child1
                # if self.himmelblau(new_child2) < self.himmelblau(child2):
                #     child2 = new_child2
            
            new_pop.append(child1)
            if len(new_pop) < len_pop:
                new_pop.append(child2)

        # Búsqueda local
        # _, new_pop = self.sort_pop(new_pop, fit)
        # if search_nearby:
        #     for i in range(len(new_pop)):
        #         new_pop[i] = self.search_nearby(new_pop[i], fit)
        
        print("final sorting...", end="")
        fitness, pop = self.sort_pop(new_pop, fit)
        return fitness, pop

    def evolve(self,
               pop: list[MLP], 
               fit: Callable[..., float], 
               pmut: float, pcross: float=0.7, 
               ngen=100, T=2, trace: int=0,
               mutate_times=1, beta=0.4, elitism=False):
        """
        Evoluciona una población durante ngen generaciones
        """
        try:
            pop = copy.deepcopy(pop)
            print("Initial population sorting...")
            fitness, pop = self.sort_pop(pop, fit)
            if trace:
                # fitness, pop = self.sort_pop(pop, fit)
                print(f"Generation    0:  (initial)  Best Fitness = {fitness[0]}")  #  f(x, y) = {self.himmelblau(pop[0])}")

            for generation in range(1, ngen+1):
                do_print = bool(trace and generation % trace == 0)
                if do_print:
                    print(f"Generation {generation:4}:  (", end="")
                start_time = time.perf_counter()
                fitness, pop = self.evolve_iteration(pop, fit, pmut, pcross, T, mutate_times, beta, elitism, do_print)
                end_time = time.perf_counter()

                # fitness, pop = self.sort_pop(pop, fit)
                self.best = pop[0]
                best_fit = fitness[0]

                if do_print:
                    time_elapsed = end_time - start_time
                    print(f")  Best Fitness = {best_fit:0.16f}  (time elapsed: {time_elapsed:4.4f}s)")  #  f(x, y) = {self.himmelblau(self.best)}")
        except KeyboardInterrupt:
            return self.best
        
        # self.pool.join()
        # self.pool.close()

        return pop[0]  # a.k.a. best

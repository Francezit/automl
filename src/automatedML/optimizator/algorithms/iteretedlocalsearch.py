import numpy as np

from .optimizehelper import OptimizationHelper


def optimize_with_iteretedlocalsearch(neiborh_size: int, num_pertubations: int,  **kargs):

    with OptimizationHelper(**kargs) as helper:

        # set current solution
        initial_individual = helper.initial_individual
        initial_fitness = None

        # run algorithm
        while not helper.is_stop_criteria_satisfy():
            helper.begin_iteration()

            population = []
            for _ in range(neiborh_size):
                individual = initial_individual.copy()
                helper.mutation(individual, num_pertubations)
                population.append(individual)

            # Calculate the fitness of each individual in the initial population
            fitness_scores = helper.evaluation(population)
            index = np.argmax(fitness_scores)
            best_fitness = fitness_scores[index]
            best_individual = population[index]
            if initial_fitness is None or best_fitness > initial_fitness:
                initial_individual = best_individual
                initial_fitness = best_fitness

            helper.end_iteration()

    # gest best metric
    return helper.get_optimization_output()


__all__ = ["optimize_with_iteretedlocalsearch"]

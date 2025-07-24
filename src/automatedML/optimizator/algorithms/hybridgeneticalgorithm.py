import random
import numpy as np

from .optimizehelper import OptimizationHelper


def optimize_with_hybridgeneticalgorithm(population_size: int, crossover_prob: float, mutation_prob: float,
                                         num_pertubations: int,  neiborh_size: int, **kargs):

    with OptimizationHelper(**kargs) as helper:

        # Initialize the population
        population = helper.get_random_individuals(population_size)

        # Calculate the fitness of each individual in the initial population
        fitness_scores = helper.evaluation(population)

        # Run the genetic algorithm for the specified number of generations
        while not helper.is_stop_criteria_satisfy():
            helper.begin_iteration()

            # Select the fittest individuals for reproduction
            parents = []
            for j in range(population_size):
                parent1 = random.choices(population, weights=fitness_scores)[0]
                parent2 = random.choices(population, weights=fitness_scores)[0]
                parents.append((parent1, parent2))

            # Create the next generation of individuals by crossover and mutation
            next_generation = []
            for j in range(population_size):
                if random.random() < crossover_prob:
                    child = helper.crossover(parents[j][0], parents[j][1])
                else:
                    child = parents[j][0].copy()
                if random.random() < mutation_prob:
                    child = helper.mutation(child, num_pertubations)
                next_generation.append(child)

            # Calculate the fitness of each individual in the population
            fitness_scores = helper.evaluation(next_generation)

            # Perform Local Search
            neiborhood = []
            best_individual: list = next_generation[np.argmax(fitness_scores)]
            for _ in range(neiborh_size):
                individual = best_individual.copy()
                helper.mutation(individual, num_pertubations)
                neiborhood.append(individual)
            neiborhood_scores = helper.evaluation(neiborhood)
            if max(neiborhood_scores) > max(fitness_scores):
                next_generation.append(neiborhood[np.argmax(neiborhood_scores)])
                fitness_scores.append(max(neiborhood_scores))

                idx = np.argmin(fitness_scores)
                del fitness_scores[idx]
                del next_generation[idx]

            # Replace the current population with the next generation
            population = next_generation
            helper.end_iteration()

    # gest best metric
    return helper.get_optimization_output()


__all__ = ["optimize_with_geneticalgorithm"]

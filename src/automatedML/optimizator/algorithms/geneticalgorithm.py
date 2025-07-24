import random

from .optimizehelper import OptimizationHelper


def optimize_with_geneticalgorithm(population_size: int, crossover_prob: float, mutation_prob: float, num_pertubations: int, **kargs):

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

            # Replace the current population with the next generation
            population = next_generation

            # Calculate the fitness of each individual in the population
            fitness_scores = helper.evaluation(population)

            helper.end_iteration()

    # gest best metric
    return helper.get_optimization_output()


__all__ = ["optimize_with_geneticalgorithm"]

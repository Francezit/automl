from .optimizehelper import OptimizationHelper


def optimize_with_localsearch(neiborh_size: int, num_pertubations: int, **kargs):

    # init helper
    with OptimizationHelper(**kargs) as helper:

        # Initialize the population
        population = []
        for _ in range(neiborh_size):
            individual = helper.initial_individual
            helper.mutation(individual, num_pertubations)
            population.append(individual)

        # Calculate the fitness of each individual in the initial population
        helper.evaluation(population)

    # gest best metric
    return helper.get_optimization_output()


__all__ = ["optimize_with_localsearch"]

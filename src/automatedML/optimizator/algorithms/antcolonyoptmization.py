import numpy as np

from .optimizehelper import OptimizationHelper
from ...utils import TupleMatrix


def optimize_with_antcolonyoptimization(n_ants: int, alpha: float, beta: float, max_path_length: int, max_neighborhood_size: int, evaporation_rate: float, Q: float, **kargs):

    with OptimizationHelper(**kargs) as helper:

        def distance(f1: float, f2: float):
            costs = [1/(1+f) for f in [f1, f2]]
            return np.abs(costs[0]-costs[1])

        pheromone = TupleMatrix(helper.search_space_size, default_value=1)

        while not helper.is_stop_criteria_satisfy():
            initial_point = helper.get_random_individual()
            initial_point_fitness = helper.evaluation(initial_point)

            paths = []
            path_lengths = []

            for k_ant in range(n_ants):

                path = [initial_point]
                path_length = 0

                current_point = initial_point
                current_point_fitness = initial_point_fitness
                while len(path) < max_path_length:

                    # neighborhoods
                    neighborhood_points = helper.neighborhood(
                        individual=current_point,
                        exclude_itself=True,
                        max_size=max_neighborhood_size)

                    n = len(neighborhood_points)
                    if n == 0:
                        helper.trace(
                            "Problem: neighborhood_points is empty!!!")
                        break

                    # precupute the fitness for each candidate, this will be used during the computation of the distance.
                    distances = np.zeros(n)
                    next_point_fintess_list = helper.evaluation(
                        neighborhood_points)
                    for i, next_point_fintess in enumerate(next_point_fintess_list):
                        distances[i] = distance(
                            current_point_fitness, next_point_fintess)

                    # compute the probabilities
                    probabilities = np.zeros(n)
                    for i, next_point in enumerate(neighborhood_points):
                        probabilities[i] = pheromone.get(current_point, next_point)**alpha /\
                            distances[i]**beta
                    probabilities /= np.sum(probabilities)

                    # select next points
                    next_point_index = np.random.choice(
                        list(range(n)), p=probabilities)
                    next_point = neighborhood_points[next_point_index]
                    path.append(next_point)
                    path_length += distances[next_point_index]

                    current_point = next_point
                    current_point_fitness = next_point_fintess_list[next_point_index]

                paths.append(path)
                path_lengths.append(path_length)
                helper.trace(f"Ant {k_ant} ends, path lenght {path_length}")

            # use evaporation
            pheromone.rescale(evaporation_rate)

            for path, path_length in zip(paths, path_lengths):
                for i in range(len(path)-1):
                    pheromone.sum(path[i], path[i+1], Q/path_length)
                pheromone.sum(path[-1], path[0], Q/path_length)

        helper.end_iteration()

    return helper.get_optimization_output()


__all__ = ["optimize_with_antcolonyoptimization"]

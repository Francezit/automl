import numpy as np

from .optimizehelper import OptimizationHelper


def optimize_with_simulatedannelaling(temp: float, step_size: int, **kargs):

    with OptimizationHelper(**kargs) as helper:
        bounds = helper.search_space_shape

        def cost_function(solution: list):
            individual = helper.convert_integers_to_individual(solution)
            fitness = helper.evaluation(individual)
            cost = 1/(1+fitness)
            return cost

        def increase_step(solution: list):
            v = solution.copy()

            n_sol = len(solution)
            steps = np.random.randint(-step_size, step_size, n_sol)
            for i in range(n_sol):
                v[i] = (v[i] + steps[i]) % bounds[i]
            return v

        # generate an initial point
        best = helper.convert_individual_to_integers(helper.initial_individual)
        # evaluate the initial point
        best_eval = cost_function(best)
        # current working solution
        curr, curr_eval = best, best_eval

        # run the algorithm
        while not helper.is_stop_criteria_satisfy():
            helper.begin_iteration()

            # take a step
            candidate = increase_step(curr)
            # evaluate candidate point
            candidate_eval = cost_function(candidate)
            # check for new best solution
            if candidate_eval < best_eval:
                # store new best point
                best, best_eval = candidate, candidate_eval

                # difference between candidate and current point evaluation
            diff = candidate_eval - curr_eval
            # calculate temperature for current epoch
            t = temp / float(helper.current_iteration + 1)
            # calculate metropolis acceptance criterion
            metropolis = np.exp(-diff / t)
            # check if we should keep the new point
            if diff < 0 or np.random.rand() < metropolis:
                # store the new current point
                curr, curr_eval = candidate, candidate_eval

            helper.end_iteration()

    return helper.get_optimization_output()

__all__=["optimize_with_simulatedannelaling"]
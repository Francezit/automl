import numpy as np

from .optimizehelper import OptimizationHelper


def optimize_with_tabuseach(tabu_list_size: int, neiborh_size: int, num_pertubations: int, **kargs):

    with OptimizationHelper(**kargs) as helper:

        sBest = helper.initial_individual
        fBest = helper.evaluation(sBest)
        tabu_list = [sBest]

        def compute_neighborhood(candidate_individual: list):
            neighborhood = []
            for _ in range(neiborh_size):
                individual = candidate_individual.copy()
                helper.mutation(individual, num_pertubations)
                neighborhood.append(individual)

            fitness = helper.evaluation(neighborhood)

            return neighborhood, fitness

        bestCandidate = helper.initial_individual
        bestCandidateFitness = -np.inf
        while not helper.is_stop_criteria_satisfy():
            helper.begin_iteration()

            sNeighborhood, fNeighborhood = compute_neighborhood(bestCandidate)
            for sCandidate, fCandidate in zip(sNeighborhood, fNeighborhood):
                if not sCandidate in tabu_list and fCandidate > bestCandidateFitness:
                    bestCandidate = sCandidate
                    bestCandidateFitness = fCandidate

            if np.isneginf(bestCandidateFitness):
                break

            if bestCandidateFitness > fBest:
                sBest = bestCandidate
                fBest = bestCandidateFitness

            tabu_list.append(bestCandidate)
            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)

            helper.end_iteration()

    return helper.get_optimization_output()


__all__ = ["optimize_with_tabuseach"]

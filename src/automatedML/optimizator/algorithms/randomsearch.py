import numpy as np

from .optimizehelper import OptimizationHelper


def optimize_with_randomsearch(**kargs):

    with OptimizationHelper(**kargs) as helper:

        while not helper.is_stop_criteria_satisfy():
            chunk = helper.get_random_individuals(helper.default_chunk_size,
                                                  unique=True)
            helper.evaluation(chunk)

    # gest best metric
    return helper.get_optimization_output()


__all__ = ["optimize_with_randomsearch"]

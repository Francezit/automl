from .optimizehelper import OptimizationHelper


def optimize_with_gridsearch(**kargs):

    with OptimizationHelper(**kargs) as helper:

        iter_search_space = helper.iter_search_space()
        chunk_size = helper.default_chunk_size

        while not helper.is_stop_criteria_satisfy():
            chunk = []
            while len(chunk) < chunk_size and iter_search_space.is_iterable:
                chunk.append(next(iter_search_space))

            if len(chunk) == 0:
                break

            helper.evaluation(chunk)

    # gest best metric
    return helper.get_optimization_output()


__all__ = ["optimize_with_gridsearch"]

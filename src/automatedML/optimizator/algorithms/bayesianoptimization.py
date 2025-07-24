import numpy as np

from .optimizehelper import OptimizationHelper


def optimize_with_bayesianopt(**kargs):
    from bayes_opt import BayesianOptimization, Events

    with OptimizationHelper(**kargs) as helper:
        bounds = helper.integers_bounds

        def objective_function(**x):
            x = helper.convert_integers_to_individual(list(x.values()))
            return helper.evaluation(x)

        params_gbm = dict(zip([f'p{x}' for x in range(0, len(bounds))], bounds))
        gbm_bo = BayesianOptimization(objective_function,
                                    params_gbm,
                                    verbose=0,
                                    random_state=1234)
        gbm_bo.subscribe(Events.OPTIMIZATION_STEP, "hpoptimize",
                        lambda e, x: helper.end_iteration())
        gbm_bo.maximize(n_iter=np.inf)

    # gest best metric
    return helper.get_optimization_output()


__all__ = ["optimize_with_bayesianopt"]

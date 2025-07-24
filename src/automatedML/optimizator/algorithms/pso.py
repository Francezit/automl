import numpy as np
from numpy import inf

from .optimizehelper import OptimizationHelper


def optimize_with_particle_swarm_optimization(n_particles: int, cognitive: float, social: float, inertia: float, **kargs):

    from pyswarms.discrete import BinaryPSO
    from pyswarms.backend import compute_objective_function, compute_pbest
    import logging
    import numpy as np
    import multiprocessing as mp
    from collections import deque

    class BinaryPSOWrapper(BinaryPSO):
        def __init__(self, **kargs):
            super().__init__(**kargs)

        def optimize(self, objective_func, stop_criterion_fun, n_processes=None, verbose=True, **kwargs):
            # Apply verbosity
            if verbose:
                log_level = logging.INFO
            else:
                log_level = logging.NOTSET

            self.rep.log("Obj. func. args: {}".format(
                kwargs), lvl=logging.DEBUG)

            # Populate memory of the handlers
            self.vh.memory = self.swarm.position

            # Setup Pool of processes for parallel evaluation
            pool = None if n_processes is None else mp.Pool(n_processes)

            self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
            ftol_history = deque(maxlen=self.ftol_iter)
            i = -1
            while not stop_criterion_fun():
                i += 1
                # Compute cost for current position and personal best
                self.swarm.current_cost = compute_objective_function(
                    self.swarm, objective_func, pool, **kwargs
                )
                self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                    self.swarm
                )
                best_cost_yet_found = np.min(self.swarm.best_cost)
                # Update gbest from neighborhood
                self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                    self.swarm, p=self.p, k=self.k
                )
                if verbose:
                    # Print to console
                    self.rep.hook(best_cost=self.swarm.best_cost)
                # Save to history
                hist = self.ToHistory(
                    best_cost=self.swarm.best_cost,
                    mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                    mean_neighbor_cost=np.mean(self.swarm.best_cost),
                    position=self.swarm.position,
                    velocity=self.swarm.velocity,
                )
                self._populate_history(hist)
                # Verify stop criteria based on the relative acceptable cost ftol
                relative_measure = self.ftol * \
                    (1 + np.abs(best_cost_yet_found))
                delta = (
                    np.abs(self.swarm.best_cost - best_cost_yet_found)
                    < relative_measure
                )
                if i < self.ftol_iter:
                    ftol_history.append(delta)
                else:
                    ftol_history.append(delta)
                    if all(ftol_history):
                        break
                # Perform position velocity update
                self.swarm.velocity = self.top.compute_velocity(
                    self.swarm, self.velocity_clamp, self.vh
                )
                self.swarm.position = self._compute_position(self.swarm)
            # Obtain the final best_cost and the final best_position
            final_best_cost = self.swarm.best_cost.copy()
            final_best_pos = self.swarm.pbest_pos[
                self.swarm.pbest_cost.argmin()
            ].copy()
            self.rep.log(
                "Optimization finished | best cost: {}, best pos: {}".format(
                    final_best_cost, final_best_pos
                ),
                lvl=log_level,
            )
            # Close Pool of Processes
            if n_processes is not None:
                pool.close()

            return (final_best_cost, final_best_pos)

    with OptimizationHelper(**kargs) as helper:

        # Set-up hyperparameters
        options = {
            'c1': cognitive, 'c2': social,
            'w': inertia,
            'k': int(np.floor(n_particles*0.3)),
            'p': 1
        }

        # Define the objective function (sphere function in this case)
        def objective_function(x: np.ndarray):
            helper.begin_iteration()
            binary_list = [y.tolist() for y in x]
            individuals = [
                helper.convert_binary_to_individual(binary)
                for binary in binary_list
            ]

            results = helper.evaluation(individuals)
            helper.end_iteration()
            return results

        # Initialize the PSO optimizer
        initial_solution = helper.convert_individual_to_binary(
            helper.initial_individual
        )
        optimizer = BinaryPSOWrapper(
            init_pos=np.array(initial_solution).reshape(
                (1, len(initial_solution))),
            n_particles=n_particles,
            options=options,
            dimensions=helper.binary_size
        )

        # Perform optimization
        optimizer.optimize(objective_function,
                           helper.is_stop_criteria_satisfy,
                           verbose=False)

    return helper.get_optimization_output()

from .localsearch import optimize_with_localsearch
from .geneticalgorithm import optimize_with_geneticalgorithm
from .hybridgeneticalgorithm import optimize_with_hybridgeneticalgorithm
from .iteretedlocalsearch import optimize_with_iteretedlocalsearch
from .bayesianoptimization import optimize_with_bayesianopt
from .gridsearch import optimize_with_gridsearch
from .randomsearch import optimize_with_randomsearch
from .tabusearch import optimize_with_tabuseach
from .antcolonyoptmization import optimize_with_antcolonyoptimization
from .gaml import optimize_with_geneticalgorithm_with_machinelearning
from .pso import optimize_with_particle_swarm_optimization
from .simulatedannealing import optimize_with_simulatedannelaling
from .optimizehelper import OptimizationResult, OptimizationCallback, OptimizationEvents,compute_hpvector_bounds


optimize_hyperparameters_methods = [
    "ls", "ga", "ils", "ts", "hga",
    "bayesian", "grid", "random",
    "aco", "gaml", "pso", "simulated_annealing"
]


def default_compare_model_function(register: list):
    import numpy as np
    return np.argmin([y["loss"] for y in register])


def optimize_hyperparameters(method: str, layers: list, ann_args: dict, train_args: dict, eval_args: dict,
                             optimize_args: dict = None, timeout: int = 300, force_instant_timeout: bool = True,
                             compare_model_function=None, train_args_bounds: dict = None, num_workers: int = None,
                             callback: OptimizationCallback = None, use_initial_random_solution: bool = False):

    if compare_model_function is None:
        compare_model_function = default_compare_model_function

    common_params = {
        "current_layers": layers,
        "ann_args": ann_args,
        "train_args": train_args,
        "eval_args": eval_args,
        "compare_model_function": compare_model_function,
        "train_args_bounds": train_args_bounds,
        "timeout": timeout,
        "num_workers": num_workers,
        "force_instant_timeout": force_instant_timeout,
        "callback": callback,
        "use_initial_random_solution": use_initial_random_solution
    }

    if optimize_args is None:
        optimize_args = {}

    if method == "ls":
        output = optimize_with_localsearch(
            **common_params,
            num_pertubations=optimize_args.get("num_pertubations", 2),
            neiborh_size=optimize_args.get("neiborh_size", 10)
        )
    elif method == "ils":
        output = optimize_with_iteretedlocalsearch(
            **common_params,
            num_pertubations=optimize_args.get("num_pertubations", 2),
            neiborh_size=optimize_args.get("neiborh_size", 5)
        )
    elif method == "ts":
        output = optimize_with_tabuseach(
            **common_params,
            num_pertubations=optimize_args.get("num_pertubations", 2),
            neiborh_size=optimize_args.get("neiborh_size", 5),
            tabu_list_size=optimize_args.get("tabu_list_size", 5)
        )
    elif method == "ga":
        output = optimize_with_geneticalgorithm(
            **common_params,
            num_pertubations=optimize_args.get("num_pertubations", 2),
            population_size=optimize_args.get("population_size", 10),
            crossover_prob=optimize_args.get("crossover_prob", 0.9),
            mutation_prob=optimize_args.get("mutation_prob", 0.4)
        )
    elif method == "hga":
        output = optimize_with_hybridgeneticalgorithm(
            **common_params,
            num_pertubations=optimize_args.get("num_pertubations", 2),
            population_size=optimize_args.get("population_size", 10),
            crossover_prob=optimize_args.get("crossover_prob", 0.9),
            mutation_prob=optimize_args.get("mutation_prob", 0.4),
            neiborh_size=optimize_args.get("neiborh_size", 5)
        )
    elif method == "gaml":
        output = optimize_with_geneticalgorithm_with_machinelearning(
            **common_params,
            num_pertubations=optimize_args.get("num_pertubations", 2),
            population_size=optimize_args.get("population_size", 10),
            crossover_params=optimize_args.get(
                "crossover_params", [0.9, 0.2, 10]),
            mutation_params=optimize_args.get(
                "mutation_params", [0.4, 0.2, 10]),
            buffer_size=optimize_args.get("buffer_size", 10)
        )
    elif method == "bayesian":
        output = optimize_with_bayesianopt(
            **common_params
        )
    elif method == "grid":
        output = optimize_with_gridsearch(
            **common_params
        )
    elif method == "random":
        output = optimize_with_randomsearch(
            **common_params
        )
    elif method == "aco":
        output = optimize_with_antcolonyoptimization(
            **common_params,
            alpha=optimize_args.get("alpha", 1),
            beta=optimize_args.get("beta", 1),
            max_neighborhood_size=optimize_args.get(
                "max_neighborhood_size", 4),
            max_path_length=optimize_args.get("max_path_length", 4),
            evaporation_rate=optimize_args.get("evaporation_rate", 0.01),
            n_ants=optimize_args.get("n_ants", 5),
            Q=optimize_args.get("Q", 1)
        )
    elif method == "pso":
        output = optimize_with_particle_swarm_optimization(
            **common_params,
            n_particles=optimize_args.get("n_particles", 10),
            cognitive=optimize_args.get("cognitive", 0.5),
            social=optimize_args.get("social", 0.3),
            inertia=optimize_args.get("inertia", 0.9)
        )
    elif method == "simulated_annealing":
        output = optimize_with_simulatedannelaling(
            **common_params,
            temp=optimize_args.get("temp", 10),
            step_size=optimize_args.get("step_size", 1)
        )
    else:
        raise Exception(f"{method} not supported")

    return output


__all__ = ["optimize_with_localsearch", "optimize_with_geneticalgorithm", "optimize_with_iteretedlocalsearch",
           "optimize_with_bayesianopt", "optimize_with_gridsearch", "optimize_with_randomsearch",
           "optimize_with_tabuseach", "optimize_with_antcolonyoptimization", "optimize_with_geneticalgorithm_with_machinelearning",
           "optimize_with_simulatedannelaling", "optimize_with_particle_swarm_optimization", "optimize_with_hybridgeneticalgorithm",
           "optimize_hyperparameters", "optimize_hyperparameters_methods", 
           "OptimizationCallback", "OptimizationEvents", "OptimizationResult","compute_hpvector_bounds"]

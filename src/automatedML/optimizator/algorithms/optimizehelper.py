import random
import numpy as np
from itertools import product
from time import time
from enum import Enum

from ...utils import SearchSpace
from ...models import BaseLayer
from ...evalmodel import evaluation_models


class OptimizationResult():

    def __init__(self, index: int, register: list, names_params: list, error_info: tuple) -> None:
        if index >= 0:
            self._index = index
            self._register = register
            self._record = register[index]
        else:
            self._index = -1
            self._register = []
            self._record = {}
        self._names_params = names_params
        self._error_info = error_info

    @property
    def has_error(self):
        return self._error_info is not None

    @property
    def is_empty(self):
        return len(self._register) == 0

    @property
    def key(self) -> str:
        return self._record["key"]

    @property
    def score(self) -> str:
        return self._record["fitness"]

    @property
    def hpvector(self) -> list:
        return self._record["raw"]

    @property
    def layers(self) -> list:
        return self._record["configuration"][0]

    @property
    def train_params(self) -> dict:
        return self._record["configuration"][1]

    @property
    def loss(self) -> float:
        return self._record["loss"]

    @property
    def other_metrics(self) -> list:
        return self._record["other_metrics"]

    @property
    def time(self):
        return self._record["time"]

    @property
    def hpvector_metadata(self) -> list:
        return self._names_params

    @property
    def n_evaluations(self) -> int:
        return len(self._register)

    @property
    def n_params(self) -> int:
        return self._record["n_params"]

    @property
    def total_duration(self) -> int:
        return self._register[-1]["time"]

    def __str__(self) -> str:
        return f"{self.key} ({self.loss}-{'-'.join(self.other_metrics)})"

    def to_dict(self):
        object_dict = {
            "key": self.key,
            "hpvector": self.hpvector,
            "hpvector_metadata": self.hpvector_metadata,
            "loss": self.loss,
            "n_evaluations": self.n_evaluations,
            "n_params": self.n_params,
            "other_metrics": self.other_metrics,
            "score": self.score,
            "time": self.time,
            "total_duration": self.total_duration
        }
        return object_dict

    def export(self, filename: str = None):
        import pandas as pd

        columns_stat = [
            "id",
            "time",
            "iteration",
            "key",
            "loss",
            "other_metrics",
            "fitness"
        ]
        columns_params = self._names_params

        table = pd.DataFrame(columns=columns_stat+columns_params)
        for id, record in enumerate(self._register):
            data = {
                "id": id,
                "time": record["time"],
                "iteration": record["generation"],
                "key": record["key"],
                "loss": record["loss"],
                "other_metrics": record["other_metrics"][0],
                "fitness": record["fitness"]
            }
            for col, item in zip(columns_params, record["raw"]):
                data[col] = item

            table.loc[len(table)] = data

        table.set_index("id", inplace=True)

        if filename is not None:
            table.to_csv(filename)
        return table

    def get_error_info(self):
        return self._error_info

    """
    def plot(self, filename: str, field: str = "fitness"):
        assert field in ["loss", "other_metrics", "fitness"]

        import matplotlib.pyplot as plt

        if field == "other_metrics":
            y = [t[field][0] for t in self._register]
        else:
            y = [t[field] for t in self._register]
        x = [t["time"] for t in self._register]

        x, y = np.array(x), np.array(y)
        i = np.argsort(x)
        x, y = x[i], y[i]

        fig, axes = plt.subplots(1, 1)
        axes.plot(x, y)
        fig.savefig(filename)
        plt.close(fig)
    """


class OptimizationEvents(Enum):
    START = "start"
    BEGIN_ITERATION = "begin-iter"
    END_ITERATION = "end-iter"
    STARTING_EVAL = "starting-eval"
    FINISHED_EVAL = "finished-eval"
    TIMEOUT = "timeout"
    ERROR = "error"
    BEST = "best"
    TRACE = "trace"


class OptimizationCallback():
    _register: dict

    def __init__(self) -> None:
        self._register = {}
        for event in (OptimizationEvents):
            self._register[event] = []

    def unsubscribe(self, event: OptimizationEvents, provider: str):
        assert event in self._register, "Event not supported"
        event_list: list = self._register[event]
        new_event_list = [x for x in event_list if x["provider"] != provider]
        self._register[event] = new_event_list

    def clear(self, provider: str = None):
        if provider is None:
            for event in self._register.keys():
                self._register[event].clear()
        else:
            for event in self._register.keys():
                self.unsubscribe(event, provider)

    def subscribe_all(self, provider: str, func):
        for event in self._register.keys():
            self.subscribe(event, provider, func)

    def subscribe(self, event: OptimizationEvents, provider: str, fun):
        if isinstance(event, list):
            for x in event:
                self.subscribe(x, provider, fun)
        else:
            assert event in self._register, "Event not supported"
            self._register[event].append({
                "provider": provider,
                "fun": fun
            })

    def call(self, event: OptimizationEvents, data: object, *args):
        assert event in self._register, "Event not supported"
        event_list: list = self._register[event]
        for item in event_list:
            f = item["fun"]
            f(event, data, *args)


class OptimizationException(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)


class OptimizationHelper():
    _register: list
    _best_index: int
    _eval_result_cache: dict
    _iteration: int
    _start_time: int
    _error: tuple
    _randState: random.Random

    _available_values: list
    _available_param_names: list
    _base_train_params: dict
    _initial_individual: list

    _initial_timeout: int
    _force_instant_timeout: bool
    _base_layers: list
    _compare_model_function: object
    _ann_args: dict
    _train_args: dict
    _eval_args: dict
    _num_workers: int
    _callback: OptimizationCallback

    def __init__(self, current_layers: list, compare_model_function, timeout: int,
                 ann_args: dict, train_args: dict, eval_args: dict,
                 num_workers: int, train_args_bounds: dict, callback: OptimizationCallback = None,
                 force_instant_timeout: bool = True, use_initial_random_solution: bool = False):

        # set params
        self._initial_timeout = timeout
        self._base_layers = current_layers
        self._compare_model_function = compare_model_function
        self._ann_args = ann_args
        self._train_args = train_args
        self._eval_args = eval_args
        self._num_workers = num_workers
        self._force_instant_timeout = force_instant_timeout
        self._callback = callback

        # init internal variables
        self._register = []
        self._eval_result_cache = {}
        self._iteration = 0
        self._start_time = time()
        self._best_index = None
        self._error = None
        self._randState = random.Random()

        # compute available_values
        v, p, t = compute_hpvector_bounds(
            current_layers,
            train_args,
            train_args_bounds
        )
        self._available_values = v
        self._available_param_names = p
        self._base_train_params = t

        # config initial solution:
        if use_initial_random_solution:
            self._initial_individual = self.get_random_individual()
        else:
            self._initial_individual = self.get_individual(
                (self._base_layers, self._base_train_params))

    @property
    def default_chunk_size(self):
        return self._num_workers if self._num_workers is not None else 4

    @property
    def current_iteration(self):
        return self._iteration

    @property
    def timeout(self):
        return self._initial_timeout - (time()-self._start_time)

    @property
    def integers_bounds(self):
        return [(0, len(x)-1) for x in self._available_values]

    @property
    def bounds(self):
        return self._available_values

    def iter_search_space(self):
        return SearchSpace(self._available_values)

    @property
    def search_space_shape(self):
        return [len(x) for x in self._available_values]

    @property
    def search_space_size(self):
        return np.prod(self.search_space_shape)

    @property
    def individual_size(self):
        return len(self._available_values)

    @property
    def binary_size(self):
        size = 0
        for values in self._available_values:
            max_integer = len(values)-1
            size += int(np.floor(np.log2(max_integer)+1))
        return size

    @property
    def initial_individual(self):
        return self._initial_individual

    @property
    def is_empty(self):
        return len(self._register) == 0

    @property
    def best_configuration(self) -> list:
        if self.is_empty:
            return None
        best_index = self._compare_model_function(self._register)
        return self._register[best_index]["configuration"]

    @property
    def best_metrics(self) -> list:
        if self.is_empty:
            return None
        best_index = self._compare_model_function(self._register)
        return [self._register[best_index]["fitness"], self._register[best_index]["loss"]] + self._register[best_index]["other_metrics"]

    def get_optimization_output(self):
        if not self.is_empty:
            return OptimizationResult(self._best_index, self._register, self._available_param_names, self._error)
        else:
            return OptimizationResult(-1, None, self._available_param_names, self._error)

    def trace(self, message: str):
        if self._callback is not None:
            self._callback.call(OptimizationEvents.TRACE, self, message)

    def notify(self, event: OptimizationEvents, *args):
        if self._callback is not None:
            self._callback.call(event, self, *args)

    def begin_iteration(self):
        self.trace(
            f"Start iteration {self.current_iteration}, available time: {self.timeout}")
        self.notify(OptimizationEvents.BEGIN_ITERATION, self._iteration)
        pass

    def end_iteration(self):
        self.trace(
            f"End iteration {self.current_iteration}, best: {', '.join([str(x) for x in self.best_metrics])}")
        self.notify(OptimizationEvents.END_ITERATION, self._iteration)
        self._iteration += 1

    def is_stop_criteria_satisfy(self) -> bool:
        return self.timeout <= 0

    def convert_individual_to_binary(self, individual: list):
        v = []
        integers = self.convert_individual_to_integers(individual)
        for i, integer in enumerate(integers):
            max_integer = len(self._available_values[i])-1

            bit_size = int(np.floor(np.log2(max_integer)+1))

            bit_string = [int(b) for b in f'{integer:b}']
            bit_array = [0 for _ in range(bit_size-len(bit_string))]+bit_string
            assert len(bit_array) == bit_size

            [v.append(x) for x in bit_array]
        return v

    def convert_binary_to_individual(self, binary: list):
        v = []
        i = 0
        for values in self._available_values:
            max_integer = len(values)-1
            bit_size = int(np.floor(np.log2(max_integer)+1))

            bit_string = ''.join([str(x) for x in list(binary[i:i+bit_size])])
            i += bit_size
            integer = int(bit_string, 2)
            v.append(integer)
        return self.convert_integers_to_individual(v)

    def convert_individual_to_integers(self, individual: list):
        v = []
        for i, x in enumerate(individual):
            v.append(self._available_values[i].index(x))
        return v

    def convert_integers_to_individual(self, integers: list):
        v = []
        for i, x in enumerate(integers):
            idx = int(np.floor(x))
            values = self._available_values[i]
            idx = idx % len(values)
            v.append(values[idx])
        return v

    def get_random_individuals(self, size: int, unique: bool = False):
        assert size <= self.search_space_size, "Size too large for the current search space"

        if not unique:
            l = [self.get_random_individual() for _ in range(size)]
            return l
        else:
            u = set()
            while len(u) < size:
                u.add(tuple(self.get_random_individual()))
            return [list(x) for x in u]

    def get_random_individual(self):
        individual = []
        for values in self._available_values:
            individual.append(
                values[self._randState.randint(0, len(values)-1)])
        return individual

    def get_individual(self, configuration: tuple):

        individual = []
        for item in configuration[0]:
            layer: BaseLayer = item
            for name in layer.get_hyperparameter_names():
                individual.append(layer.get_hyperparameter(name))

        _extract_train_values(configuration[1], individual)
        return individual

    def get_random_configuration(self):
        individual = self.get_random_individual()
        return self.get_configuration(individual)

    def get_configuration(self, individual: list):
        layers = []
        index = 0
        for item in self._base_layers:
            layer: BaseLayer = item.clone()
            for name in layer.get_hyperparameter_names():
                layer.set_hyperparameter(name, individual[index])
                index += 1
            layers.append(layer)

        train_params = _import_train_value(
            self._base_train_params, individual[index:])
        return layers, train_params

    def crossover(self, parent1: list, parent2: list):
        # Perform a crossover operation between two parents to create a new child
        # For example, we'll randomly select genes from either parent
        child = []
        for i in range(len(parent1)):
            if self._randState.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def neighborhood(self, individual: list, indices: list = None, exclude_itself: bool = False, max_size: int = None):
        if indices is None:
            indices = [self._randState.randint(0, len(individual)-1)]
        elif isinstance(indices, int):
            indices = [indices]

        item_list = []
        availables = [self._available_values[index] for index in indices]
        combination_list = product(*availables)
        for pair_value in combination_list:
            candidate = individual.copy()
            for i, index in enumerate(indices):
                candidate[index] = pair_value[i]
            item_list.append(candidate)

        if exclude_itself:
            item_list.remove(individual)

        if max_size is not None and len(item_list) >= max_size:
            for _ in range(len(item_list)-max_size):
                item_list.pop()

        return item_list

    def mutation(self, individual: list, num_pertubations: int):
        # Perform a mutation operation on an individual to introduce random changes
        for _ in range(num_pertubations):
            index = self._randState.randint(0, len(individual)-1)
            values = self._available_values[index]
            individual[index] = values[self._randState.randint(
                0, len(values)-1)]
        return individual

    def evaluation(self, population: list):

        direct_output = False

        if len(population) == 0:
            return []
        elif isinstance(population[0], list) or isinstance(population[0], tuple):
            n = len(population[0])
            assert all([len(item) == n for item in population]
                       ), "Each individual must have the same size"
        elif isinstance(population, SearchSpace):
            pass
        else:
            population = [population]
            direct_output = True

        # set internal variables
        n_params = len(population[0])
        nn_configs = [self.get_configuration(x) for x in population]
        nn_keys = ['-'.join([str(y) for y in x]) for x in population]
        nn_results = [None for _ in range(len(nn_configs))]

        # notify
        self.trace(f"Evaluating {len(nn_configs)} models")
        self.notify(OptimizationEvents.STARTING_EVAL, nn_configs)

        # handle cache
        nn_key_test = []
        nn_configs_test = []
        nn_result_test_indices = []
        for index, key in enumerate(nn_keys):
            result = self._eval_result_cache.get(key, None)
            if result is None:
                if not key in nn_key_test:
                    nn_key_test.append(key)
                    nn_configs_test.append(nn_configs[index])
                    nn_result_test_indices.append([index])
                else:
                    idx = nn_key_test.index(key)
                    nn_result_test_indices[idx].append(index)
            else:
                nn_results[index] = result

        # test new configurations
        if len(nn_configs_test) > 0:
            nn_results_test = evaluation_models(configurations=nn_configs_test,
                                                ann_args=self._ann_args,
                                                train_args=self._train_args,
                                                eval_args=self._eval_args,
                                                num_workers=self._num_workers,
                                                timeout=self.timeout,
                                                force_instant_timeout=self._force_instant_timeout)
        else:
            nn_results_test = []

        # update configuration
        for indices, result in zip(nn_result_test_indices, nn_results_test):
            for index in indices:
                nn_results[index] = result

        # compute score
        fitness_scores = []
        for raw, key, config, result in zip(population, nn_keys, nn_configs, nn_results):
            if not key in self._eval_result_cache:
                self._eval_result_cache[key] = result

            if np.isfinite(result["loss"]):
                fitness = 1/(result["loss"] + 1)
            else:
                fitness = 0

            self._register.append({
                "key": key,
                "raw": raw,
                "configuration": config,
                "loss": result["loss"],
                "other_metrics": result["other_metrics"],
                "fitness": fitness,
                "generation": self._iteration,
                "n_params": n_params,
                "time": time()-self._start_time
            })
            fitness_scores.append(fitness)

        # trace information
        self.trace(
            f"Evaluated {len(fitness_scores)} models, best fitness={np.max(fitness_scores)}")

        # select best
        best_index: int = self._compare_model_function(self._register)
        if self._best_index is None or self._best_index != best_index:
            self._best_index = best_index
            self.notify(OptimizationEvents.BEST,
                        self.get_optimization_output())

        # notify ends
        self.notify(OptimizationEvents.FINISHED_EVAL,
                    self.get_optimization_output())

        # check budget time
        if self.is_stop_criteria_satisfy():
            self.trace("Event timeout occured")
            raise OptimizationException()

        return fitness_scores if not direct_output else fitness_scores[0]

    def __enter__(self):
        self.notify(OptimizationEvents.START)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.is_stop_criteria_satisfy():
            self.notify(OptimizationEvents.TIMEOUT,
                        self.get_optimization_output())
            return True
        elif exc_value is not None:
            self.notify(OptimizationEvents.ERROR,
                        str(exc_value),
                        str(exc_traceback))
            self._error = (str(exc_value), str(exc_traceback))
        return True


def compute_hpvector_bounds(layers: list[BaseLayer], train_args: dict, train_args_bounds: dict):
    available_param_names = []
    available_values = []

    for item in layers:
        layer: BaseLayer = item
        for name in layer.get_hyperparameter_names():
            hp_format = layer.get_hyperparameter_format(name)
            available_values.append(hp_format.get_discrete_values())
            available_param_names.append(name)

    # add train hyperparameters
    base_train_params = {}
    if train_args_bounds is not None:
        for train_hp_key in train_args_bounds.keys():
            assert train_hp_key in train_args, f"train_hp={train_hp_key} not valid"

            template = train_args_bounds[train_hp_key]
            __find_train_params(template,
                                train_hp_key,
                                available_values,
                                available_param_names)
            base_train_params[train_hp_key] = __update_train_params(template,
                                                                    train_args[train_hp_key])

    # check params
    n_params = len(available_values)
    assert n_params > 0, "Architecture without any hyperparamenters"

    # handle names
    for i in range(n_params):
        test = available_param_names[i]

        indices = map(lambda e, i: i if e == test else -1,
                      available_param_names, range(n_params))
        indices = filter(lambda e: e >= 0, indices)
        indices = list(indices)
        if len(indices) >= 2:
            for id, idx in enumerate(indices):
                available_param_names[idx] = f"{available_param_names[idx]}_{id+1}"

    return available_values, available_param_names, base_train_params


def _import_train_value(template, input: list):
    if isinstance(template, dict):
        obj = {}
        for key in template:
            obj[key] = _import_train_value(template[key], input)
        return obj
    else:
        return input.pop(0)


def _extract_train_values(template, output: list):
    if isinstance(template, dict):
        for key in template:
            _extract_train_values(template[key], output)
    else:
        output.append(template)


def __find_train_params(template, key: str, output: list, output_names: list):
    if isinstance(template, list):
        output.append(template)
        output_names.append(key)
    elif isinstance(template, dict):
        for template_key in template:
            __find_train_params(template[template_key],
                                f"{key}_{template_key}",
                                output,
                                output_names)


def __update_train_params(template, value):
    if isinstance(template, list):
        if value is None:
            return template[0]
        else:
            assert value in template, f"hp ({value}) not supported"
            return value
    elif isinstance(template, dict):
        obj = {}
        for key in template:
            obj[key] = __update_train_params(template[key],
                                             value[key] if value is not None and isinstance(value, dict) and key in value else None)
        return obj
    else:
        raise Exception("update_train_params error")


__all__ = ["OptimizationResult", "OptimizationEvents", "OptimizationCallback",
           "OptimizationHelper", "compute_hpvector_bounds"]

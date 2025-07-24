import numpy as np
import queue
import multiprocessing
import threading
import os
from time import time

from .internal import get_number_metrics, advanced_dic_update
from .models import BaseLayer


def evaluation_models(configurations: list[list[BaseLayer] | tuple[list[BaseLayer, dict]]],
                      ann_args: dict, train_args: dict, eval_args: dict,
                      num_workers: int = None, timeout: float = None, force_instant_timeout: bool = True):

    # set variables
    nn_configs = len(configurations)
    if nn_configs == 0:
        return []

    # check configurations:
    for i in range(nn_configs):
        if isinstance(configurations[i], tuple):
            assert len(
                configurations[i]) == 2, "format of configurations not supported"
        else:
            configurations[i] = configurations[i], {}

    # set number workers
    if num_workers is None:
        num_workers = os.cpu_count()
    num_workers = min(num_workers, nn_configs)

    # define timeout mode
    if force_instant_timeout:
        worker_timeout = timeout
        internal_timeout = None
    else:
        worker_timeout = None
        internal_timeout = timeout

    type_of_task = ann_args.get("type_of_task")
    n_outputs_for_config = 1+get_number_metrics(type_of_task)

    # set results list
    objective_results: list = multiprocessing.Array(
        'f',
        [np.inf for _ in range(nn_configs*n_outputs_for_config)]
    )

    # init task queue
    task_args = [
        {
            "objective_args": {
                "layers": config[0],
                "ann_args": ann_args,
                "train_args": advanced_dic_update(train_args, config[1]),
                "eval_args": eval_args
            },
            "result": objective_results,
            "id": id*n_outputs_for_config
        }
        for id, config in enumerate(configurations)
    ]
    task_queue = queue.Queue(len(task_args))
    [task_queue.put(x) for x in task_args]
    del task_args

    # process tasks
    if nn_configs > 1 and num_workers > 1:

        # init workers
        worker_list = []
        for worker_index in range(num_workers):
            p = threading.Thread(
                target=__worker_fun,
                kwargs={
                    "worker_tasks": task_queue,
                    "worker_timeout": worker_timeout
                },
                daemon=True,
                name=f"eval_worker_{worker_index}"
            )
            worker_list.append(p)

        # start workers
        for worker in worker_list:
            worker.start()

        # join workers
        timeout_occurred = False
        for w in worker_list:
            worker: threading.Thread = w

            st = time()
            worker.join(internal_timeout)
            if internal_timeout is not None:
                internal_timeout = internal_timeout-(time()-st)
                if internal_timeout <= 0 or worker.is_alive():
                    timeout_occurred = True
                    break

        # check if timeout occurred
        if timeout_occurred:
            with task_queue.mutex:
                task_queue.queue.clear()
            for w in worker_list:
                worker: threading.Thread = w
                if worker.is_alive():
                    worker.join()

        # deallocate resources
        del worker_list

    else:
        __worker_fun(task_queue, worker_timeout)

    # process results
    results = []
    i, j = 0, 0
    while i < nn_configs:
        loss = objective_results[j]
        if n_outputs_for_config > 2:
            metrics = objective_results[j+1:j+n_outputs_for_config-1]
        else:
            metrics = [objective_results[j+1]]

        results.append({
            "loss": loss,
            "other_metrics": metrics
        })
        i += 1
        j += n_outputs_for_config
    del objective_results
    return results


def __worker_fun(worker_tasks: queue.Queue, worker_timeout: float):
    totale_time = worker_timeout

    while not worker_tasks.empty() and (totale_time is None or totale_time > 0):
        st = time()

        # extract data
        args = worker_tasks.get()

        # lunch process
        p = multiprocessing.Process(target=__process_fun,
                                    kwargs=args)
        p.start()
        p.join(totale_time)

        # check if a event of timeout occured
        if p.is_alive():
            p.kill()
            break

        # update available time
        if totale_time is not None:
            totale_time = totale_time-(time()-st)
    pass


def __process_fun(objective_args: dict, result: list, id: int):
    loss, metrics = __objective_function(**objective_args)
    result[id] = loss
    if isinstance(metrics, list):
        for i, m in enumerate(metrics):
            result[id+1+i] = m


def __objective_function(layers: list[BaseLayer], ann_args: dict, train_args: dict, eval_args: dict):
    from .ann import create_ann_by_model

    loss, other_metrics = np.inf, [np.inf]
    try:
        model = create_ann_by_model(
            model=layers,
            raise_exception=True,
            **ann_args
        )
        if model is not None:
            train_loss, _ = model.train(**train_args)
            loss, other_metrics = model.eval(**eval_args)
        else:
            print("Error in evaluating model")
        del model
    except Exception as err:
        print(err)
    return loss, other_metrics


__all__ = ["evaluation_models"]

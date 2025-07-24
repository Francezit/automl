import numpy as np
import logging
import time
import os
import json

from .knowledge import KnowledgeGraphInterface, EmptyKnowledgeGraph, KnowledgeGraph

from ..component import BaseComponent, BaseComponentSettings
from ..datacontainer import DataContainer
from ..utils import StatisticInfo, MetricInfo, stop_criterion, select_best_metric, call_function_begin_logger, call_function_end_logger, trace_function_logger, call_function_end_error_logger
from ..models import BaseLayer, get_layer_by_metadata, compute_layermap, get_statistic_from_layermap, HyperparametersManager
from ..ann import ANNModel,ANNArchitecture
from ..optimizator import optimize_hyperparameters, OptimizationCallback, OptimizationEvents
from ..internal import get_problem_type
from ..evalmodel import evaluation_models


class FlatGeneratorSettings(BaseComponentSettings):
    def __init__(self, settings: dict = None) -> None:
        super().__init__(settings)

    layer_classes: list[str] = None

    alg_max_episodes: int = 100
    alg_hp_optimization_prob: float = 0.5
    alg_enable_model_versioning: bool = False

    knowledgegraph_enable_versioning: bool = False
    knowledgegraph_reward_options: dict = None

    dyann_max_deph_size: int = 10
    dyann_max_trainable_parameters: int = 100000
    dyann_cost_metric_coeff: float = 0.7
    dyann_cost_density_coeff: float = 0.1
    dyann_cost_dephIndex_coeff: float = 0.2
    dyann_max_freeze_error: int = 3

    dyann_stop_criterion_fun: str = "probabilistic"
    dyann_stop_criterion_args: list = [0.9]

    hp_optimization_alg: str = "ls"
    hp_optimization_alg_params: dict = None
    hp_optimization_number_workers: int = None
    hp_optimization_timeout: int = 300


class FlatGenerator(BaseComponent):
    _settings: FlatGeneratorSettings

    _layer_map: dict
    _context_name: str

    def __init__(self,
                 data: DataContainer,
                 settings: FlatGeneratorSettings,
                 hp_manager: HyperparametersManager,
                 logger: logging.Logger = None,
                 context_name: str = None,
                 seed: int = None):

        super().__init__(
            data=data,
            settings=settings,
            hp_manger=hp_manager,
            logger=logger,
            seed=seed
        )

        self._context_name = context_name
        self._layer_map = compute_layermap(
            self._hp_manager,
            settings.layer_classes
        )
        stat_layer_map = get_statistic_from_layermap(
            self._layer_map, self._hp_manager
        )
        self._logger.info(f"\n{stat_layer_map}")

    def _extract_metric(self, model: ANNModel, loss: float, other_metrics: list) -> MetricInfo:
        # compute cost
        n_parameters = model.number_parameters
        n_layers = model.number_hidden_layers
        density = n_parameters/n_layers
        deepIndex = n_layers/self._settings.dyann_max_deph_size

        density_max = self._settings.dyann_max_trainable_parameters / \
            self._settings.dyann_max_deph_size
        density_factor = density/density_max

        cost = (loss*self._settings.dyann_cost_metric_coeff) +\
            (self._settings.dyann_cost_density_coeff*density_factor) + \
            (self._settings.dyann_cost_dephIndex_coeff*deepIndex)

        return MetricInfo(cost, loss, density, deepIndex, other_metrics)

    def _train_and_eval_model(self, model: ANNModel,  use_separate_process: bool = False, use_fast_train: bool = True) -> MetricInfo:

        # get parameters
        train_params, eval_params = self._compute_train_and_eval_params(
            "fast" if use_fast_train else "default"
        )

        if use_separate_process:
            results = evaluation_models(configurations=[model.get_layers()],
                                        ann_args=self._ann_base_params,
                                        train_args=train_params,
                                        eval_args=eval_params,
                                        num_workers=1)
            return self._extract_metric(model, results[0]["loss"], results[0]["other_metrics"])
        else:
            model.train(**train_params)
            loss, other_metrics = model.eval(**eval_params)

            return self._extract_metric(model, loss, other_metrics)

    def optimize_hyperparameters(self, method: str, model: ANNModel, use_fast_train: bool):
        method = method.lower()

        logId = call_function_begin_logger(
            nameFunction=f"optimize_hyperparameters_{method}",
            fun=self._logger.info,
            context_name=self._context_name
        )

        train_params, eval_params = self._compute_train_and_eval_params(
            "fast" if use_fast_train else "default"
        )

        layers: list
        if isinstance(model, ANNModel):
            layers = model.get_layers()
        elif isinstance(model, list):
            layers = model
        else:
            raise Exception("Model not valid")

        callback = OptimizationCallback()

        def handle_trace(event, sender, message):
            trace_function_logger(
                logId=logId,
                msg=message,
                fun=self._logger.info
            )
        callback.subscribe(OptimizationEvents.TRACE, "generator", handle_trace)

        opt_output = optimize_hyperparameters(
            method=method,
            layers=layers,
            ann_args=self._ann_base_params,
            train_args=train_params,
            eval_args=eval_params,
            optimize_args=self._settings.hp_optimization_alg_params,
            force_instant_timeout=False,
            timeout=self._settings.hp_optimization_timeout,
            callback=callback,
            num_workers=self._settings.hp_optimization_number_workers
        )
        opt_model = ANNModel(**self._ann_base_params)
        opt_model.make(opt_output.layers)

        best_metric: MetricInfo = self._extract_metric(model=opt_model,
                                                       loss=opt_output.loss,
                                                       other_metrics=opt_output.other_metrics)

        call_function_end_logger(
            logId=logId,
            fun=self._logger.info
        )

        return opt_model, best_metric

    def fit(self, annmodel: ANNModel):
        logId = call_function_begin_logger(
            nameFunction="fit",
            fun=self._logger.info,
            context_name=self._context_name
        )

        annmodel.edit()
        annmodel.freeze()
        metric: MetricInfo = self._train_and_eval_model(model=annmodel,
                                                        use_separate_process=False,
                                                        use_fast_train=False)

        call_function_end_logger(
            logId=logId,
            fun=self._logger.info
        )

        return metric

    def export(self, name: str, annmodel: ANNModel, estimate_cost: MetricInfo, output_folder: str):
        logId = call_function_begin_logger(
            nameFunction="export",
            fun=self._logger.info,
            context_name=self._context_name
        )

        try:
            annmodel.edit()
            if annmodel.freeze():
                metric_info: MetricInfo = self._train_and_eval_model(
                    model=annmodel,
                    use_separate_process=False,
                    use_fast_train=False
                )
                filename = os.path.join(output_folder, f"{name}")
                annmodel.save(filename)
            else:
                metric_info = MetricInfo.empty()

            with open(os.path.join(output_folder, f"{name}_metric_info.json"), "w") as fp:
                json.dump({
                    "simple_metric": estimate_cost.to_dict(),
                    "full_metric": metric_info.to_dict()
                }, fp)

            call_function_end_logger(
                logId=logId,
                fun=self._logger.info
            )

            return metric_info

        except Exception as err:
            call_function_end_error_logger(
                logId=logId,
                err=str(err),
                fun=self._logger.error
            )
            return MetricInfo.empty()

    def generate(self, knowledgegraph: KnowledgeGraphInterface):
        logId = call_function_begin_logger(
            nameFunction="generate",
            fun=self._logger.info,
            context_name=self._context_name
        )

        currentANN = ANNModel(**self._ann_base_params)

        metricHistory = []
        costHistory = []
        layerHistory = []
        countfreezeError = 0
        countLayers = 0
        while countfreezeError < self._settings.dyann_max_freeze_error and countLayers < self._settings.dyann_max_deph_size and not stop_criterion(method=self._settings.dyann_stop_criterion_fun,
                                                                                                                                                   cost_history=costHistory,
                                                                                                                                                   args=self._settings.dyann_stop_criterion_args):

            # select layer
            layerMetadata = knowledgegraph.choice_next_layer(
                prev_metadata_sequence=layerHistory
            )
            layer: BaseLayer = get_layer_by_metadata(metadata=layerMetadata,
                                                     hp_manager=self._hp_manager)

            # build ann
            currentANN.edit()
            currentANN.add_layers(layer)

            trace_function_logger(
                logId=logId,
                msg=f"Adding {layer.layerKey}",
                fun=self._logger.info
            )

            if currentANN.freeze():  # valid layer
                countfreezeError = 0
                countLayers = countLayers + 1
                metricInfo = self._train_and_eval_model(model=currentANN,
                                                        use_separate_process=True,
                                                        use_fast_train=True)

                trace_function_logger(
                    logId=logId,
                    msg=f"Added {layer.layerKey}: {metricInfo})",
                    fun=self._logger.info
                )
            else:  # not valid layer
                countfreezeError = countfreezeError+1
                metricInfo = MetricInfo.empty()
                currentANN.remove_layer(layerIndex=-1)

                trace_function_logger(
                    logId=logId,
                    msg=f"Added a not valid layer {layer.layerKey}: {currentANN.freeze_error}",
                    fun=self._logger.info
                )

            costHistory.append(metricInfo.cost)
            layerHistory.append(layer.metadata)
            metricHistory.append(metricInfo)

        knowledgegraph.register(
            metadata_sequence=layerHistory,
            fitness=costHistory
        )

        # extract best ann
        bestIndex: int = select_best_metric(metrics=metricHistory,
                                            field_name="other_metrics",
                                            type=get_problem_type(self._data.type_of_task))
        metric: MetricInfo = metricHistory[bestIndex]
        layers = []
        for i in range(min(bestIndex+1, len(costHistory))):
            cost = costHistory[i]
            if np.isfinite(cost):
                layer = get_layer_by_metadata(metadata=layerHistory[i],
                                              hp_manager=self._hp_manager)
                layers.append(layer)

        if len(layers) > 0:
            currentANN.clear()
            currentANN.make(layers)

            call_function_end_logger(
                logId=logId,
                fun=self._logger.info
            )
        else:
            currentANN.clear()
            metric = MetricInfo.empty()

            call_function_end_error_logger(
                logId=logId,
                err="I cannot find a valid network",
                fun=self._logger.error
            )

        return currentANN, metric

    def generate_randomly(self):
        knowledgegraph = EmptyKnowledgeGraph(self._layer_map)
        annmodel, metric = self.generate(knowledgegraph)
        return annmodel, metric

    def generate_iteratively(self, output_folder: str = None, knowledgegraph: KnowledgeGraphInterface = None):

        logId = call_function_begin_logger(
            nameFunction="generate_iteratively",
            fun=self._logger.info,
            context_name=self._context_name
        )

        has_output = output_folder is not None
        if has_output:
            os.makedirs(output_folder,exist_ok=True)

        # init knowledgegraph
        if knowledgegraph is None:
            knowledgegraph = KnowledgeGraph(self._layer_map,
                                            self._settings.knowledgegraph_reward_options)

        knowledge_filename: str = None
        knowledge_item_filename: str = None
        knowledge_class_filename: str = None
        if has_output:
            knowledge_filename = os.path.join(output_folder,
                                              "knowledgegraph.json")
            knowledge_item_filename = os.path.join(output_folder,
                                                   "knowledgegraph_itemgraph.svg")
            knowledge_class_filename = os.path.join(output_folder,
                                                    "knowledgegraph_classgraph.svg")

        # init stats
        stats: StatisticInfo = None
        if has_output:
            stats = StatisticInfo(os.path.join(output_folder, "stat.csv"))

        # init knowledge versioning
        knowledge_versioning_folder: str = None
        if has_output and self._settings.knowledgegraph_enable_versioning:
            knowledge_versioning_folder = os.path.join(output_folder,
                                                       "versioning-knowledge")
            os.makedirs(knowledge_versioning_folder, exist_ok=True)

        # init model versioning
        model_versioning_folder: str = None
        if has_output and self._settings.alg_enable_model_versioning:
            model_versioning_folder = os.path.join(output_folder,
                                                   "model-versioning")
            os.makedirs(model_versioning_folder, exist_ok=True)

        # start episodes
        n_episode = 0
        best_ANN: ANNModel = None
        best_metric: MetricInfo = None

        while (n_episode < self._settings.alg_max_episodes):
            st = time.time()

            # create a new artificial neural network
            init_ANN, init_metric = self.generate(knowledgegraph)
            trace_function_logger(
                logId=logId,
                msg=f"[{n_episode+1} of {self._settings.alg_max_episodes}] generateDynamicANN: {init_metric}",
                fun=self._logger.info
            )

            # use hyperparameter optimization with a probability in order to improve the hyperparameter
            if not init_ANN.is_empty and np.random.rand() <= self._settings.alg_hp_optimization_prob:

                opt_ANN, opt_metric = self.optimize_hyperparameters(method=self._settings.hp_optimization_alg,
                                                                    model=init_ANN,
                                                                    use_fast_train=True)

                optimizeSuccess = opt_metric < init_metric if opt_metric.is_valid else False

                trace_function_logger(
                    logId=logId,
                    msg=f"[{n_episode+1} of {self._settings.alg_max_episodes}] optimizeHyperparameter: {opt_metric} ({optimizeSuccess})",
                    fun=self._logger.info
                )
            else:
                opt_ANN = None
                opt_metric = MetricInfo.empty()
                optimizeSuccess = False

            # compute episode solution
            if optimizeSuccess:
                episode_ANN = opt_ANN
                episode_metric = opt_metric
                knowledgegraph.update(
                    metadata_sequence=opt_ANN.get_layer_metadatas(),
                    fitness=opt_metric.cost
                )
            else:
                episode_ANN = init_ANN
                episode_metric = init_metric

            trace_function_logger(
                logId=logId,
                msg=f"[{n_episode+1} of {self._settings.alg_max_episodes}] episodeSolution: {episode_metric}",
                fun=self._logger.info
            )

            # check current solution
            if best_metric is None or episode_metric <= best_metric:
                best_ANN = episode_ANN
                best_metric = episode_metric

            trace_function_logger(
                logId=logId,
                msg=f"[{n_episode+1} of {self._settings.alg_max_episodes}] globalSolution: {best_metric}",
                fun=self._logger.info
            )

            # update episode
            n_episode = n_episode+1
            et = time.time()

            # save epoch model
            full_metric: MetricInfo = MetricInfo.empty()
            if model_versioning_folder:
                full_metric = self.export(name=f"E{n_episode}",
                                          annmodel=episode_ANN,
                                          estimate_cost=episode_metric,
                                          output_folder=model_versioning_folder)

            # save knowledge
            if knowledge_versioning_folder:
                knowledgegraph.save(os.path.join(knowledge_versioning_folder,
                                                 f"E{n_episode}.json"))

            if knowledge_filename:
                knowledgegraph.save(knowledge_filename)

            if knowledge_class_filename:
                knowledgegraph.saveplot(filename=knowledge_class_filename,
                                        extended_version=False)

            if knowledge_item_filename:
                knowledgegraph.saveplot(filename=knowledge_item_filename,
                                        extended_version=True)

            # save stats
            if stats:
                stats.append(initMetric=init_metric,
                             optMetric=opt_metric,
                             episodeMetric=episode_metric,
                             bestMetric=best_metric,
                             fullMetric=full_metric,
                             optimizeSuccess=optimizeSuccess,
                             episode_time=et-st)

        call_function_end_logger(
            logId=logId,
            fun=self._logger.info
        )

        return best_ANN, best_metric

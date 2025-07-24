import logging
import json

from ..component import BaseComponent, BaseComponentSettings, TrainSettings
from ..datacontainer import DataContainer
from ..models import parse_layers, BaseLayer, HyperparametersManager
from ..ann import check_layer_sequence, ANNArchitecture
from ..utils import call_function_begin_logger, call_function_end_logger, trace_function_logger, call_function_end_error_logger

from .algorithms import optimize_hyperparameters, OptimizationEvents, OptimizationCallback, OptimizationResult


class OptimizatorSettings(BaseComponentSettings):
    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

    train_settings_bounds: dict = None
    use_fast_train: bool = False
    timeout: int = 300
    force_instant_timeout: bool = False
    num_workers: int = None


def load_optimizator_settings(filename: str)->OptimizatorSettings:
    with open(filename, "r") as fp:
        dt = json.load(fp)
        return OptimizatorSettings(dt)


class HyperparametersOptimizator(BaseComponent):
    _settings: OptimizatorSettings
    __layers: list[BaseLayer]

    def __init__(self,
                 model: list[BaseLayer | str] | ANNArchitecture,
                 data: DataContainer,
                 settings: OptimizatorSettings,
                 hp_manager: HyperparametersManager,
                 logger: logging.Logger = None,
                 seed: int = None):

        super().__init__(
            data=data,
            settings=settings,
            hp_manger=hp_manager,
            logger=logger,
            seed=seed
        )

        # set layers
        if isinstance(model, ANNArchitecture):
            self.__layers = model.to_model()
        else:
            self.__layers = parse_layers(model, self._hp_manager)

        valid, status = check_layer_sequence(
            self.__layers, **self._ann_base_params
        )
        if not valid:
            raise Exception(f"Sequence of layer not valid: {status}")

        # set callback
        self._callback = OptimizationCallback()

    def optimize(self, alg_name: str, stat_filename: str = None, callbackFn=None,  **kargs):

        logId = call_function_begin_logger(
            nameFunction=f"optimize",
            fun=self._logger.info
        )

        # get parameters
        train_params, eval_params = self._compute_train_and_eval_params(
            "fast" if self._settings.use_fast_train else "default"
        )

        # set callback
        PROVIDER_NAME = "optimizator"
        callback = OptimizationCallback()

        # custom callback
        if callbackFn:
            callback.subscribe_all(
                provider="external",
                fun=callbackFn
            )

        # write in the logger
        def handle_trace(event, sender, message: str):
            trace_function_logger(
                logId=logId,
                msg=message,
                fun=self._logger.info
            )
        callback.subscribe(
            event=OptimizationEvents.TRACE,
            provider=PROVIDER_NAME,
            fun=handle_trace
        )

        # save statistic
        if stat_filename:
            def handle_update(event, sender, result: OptimizationResult):
                result.export(stat_filename)

            callback.subscribe(
                event=OptimizationEvents.FINISHED_EVAL,
                provider=PROVIDER_NAME,
                fun=handle_update
            )

        # optimize
        opt_output = optimize_hyperparameters(
            method=alg_name,
            layers=self.__layers,
            ann_args=self._ann_base_params,
            train_args=train_params,
            eval_args=eval_params,
            optimize_args=kargs,
            timeout=self._settings.timeout,
            force_instant_timeout=self._settings.force_instant_timeout,
            train_args_bounds=self._settings.train_settings_bounds,
            num_workers=self._settings.num_workers,
            use_initial_random_solution=True,
            callback=callback
        )

        # handle error
        if opt_output.has_error:
            errs = [str(x) for x in opt_output.get_error_info()]
            call_function_end_error_logger(
                logId=logId,
                err=f"{'-'.join(errs)}",
                fun=self._logger.error
            )
        else:
            call_function_end_logger(
                logId=logId,
                fun=self._logger.info
            )

        if stat_filename:
            opt_output.export(stat_filename)

        return opt_output

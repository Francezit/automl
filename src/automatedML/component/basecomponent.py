import numpy as np
import logging

from .settings import BaseSettings

from ..datacontainer import DataContainer, downsample_data, split_data
from ..models import HyperparametersManager


class TrainSettings(BaseSettings):
    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

    fast_train_data_reduction_factor: float = 1
    fast_train_epochs: int = None

    epochs: int = 5
    shuffle: bool = False
    batch_size: int = None
    optimizer: str = "adam"
    optimizer_configs: dict = {}

    num_workers: int = 4
    loss_function_name: str = None
    use_early_stopping: bool = True
    validation_split: float = None


class BaseComponentSettings(BaseSettings):

    train_settings: TrainSettings = TrainSettings()

    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)


class BaseComponent():
    _data: DataContainer
    _logger: logging.Logger
    _settings: BaseComponentSettings
    _seed: int

    _hp_manager: HyperparametersManager
    _ann_base_params: dict
    _train_and_eval_params_cache: dict

    def __init__(self,
                 data: DataContainer,
                 settings: BaseComponentSettings,
                 hp_manger: HyperparametersManager,
                 logger: logging.Logger = None,
                 seed: int = None) -> None:

        assert data, "data not valid"
        assert settings, "settings not valid"
        assert hp_manger, "hp manager not valid"

        self._data = data
        self._settings = settings
        self._logger = logger if logger is not None else logging.getLogger()
        self._seed = seed if seed is not None else np.random.randint(0, 1000)
        self._hp_manager = hp_manger

        self._ann_base_params = {
            "input_size": self._data.input_size,
            "output_size": self._data.output_size,
            "type_of_task": self._data.type_of_task,
            "seed": self._seed
        }
        self._train_and_eval_params_cache = {}

    def _compute_train_and_eval_params(self, type: str):

        if type in self._train_and_eval_params_cache:
            return self._train_and_eval_params_cache[type]
        else: 
            train_params, eval_params= compute_train_and_eval_params(
                data=self._data,
                train_settings=self._settings.train_settings
            )

            self._train_and_eval_params_cache[type] = (
                train_params, eval_params
            )
            return train_params, eval_params


def compute_train_and_eval_params(data: DataContainer, train_settings: TrainSettings,**kargs) -> tuple[dict, dict]:

    train_params = {
        "batch_size": train_settings.batch_size,
        "epochs": train_settings.epochs,
        "optimizer": train_settings.optimizer,
        "optimizer_configs": train_settings.optimizer_configs,
        "shuffle": train_settings.shuffle,
        "num_workers": train_settings.num_workers,
        "loss_function_name": train_settings.loss_function_name,
        "use_early_stopping": train_settings.use_early_stopping,
        "use_tensor_board": None
    }
    train_params.update(kargs)

    X_train, y_train = data.get_trainingset()
    X_test, y_test = data.get_testset()

    if train_settings.validation_split is not None and train_settings.validation_split > 0 and train_settings.validation_split < 1:
        x_new_train, y_new_train, x_val, y_val = split_data(
            x=X_train,
            y=y_train,
            split=train_settings.validation_split
        )

        X_train = x_new_train
        y_train = y_new_train
        train_params["X_validation"] = x_val
        train_params["y_validation"] = y_val

    if type == "fast":
        if train_settings.fast_train_epochs is not None:
            train_epochs = min(train_settings.fast_train_epochs,
                               train_settings.epochs)

        if train_settings.fast_train_data_reduction_factor is not None and train_settings.fast_train_data_reduction_factor < 1:
            X_train, y_train, _ = downsample_data(x=X_train,
                                                  y=y_train,
                                                  type_of_task=data.type_of_task,
                                                  reduction_factor=train_settings.fast_train_data_reduction_factor,
                                                  seed=1234)

            X_test, y_test, _ = downsample_data(x=X_test,
                                                y=y_test,
                                                type_of_task=data.type_of_task,
                                                reduction_factor=train_settings.fast_train_data_reduction_factor,
                                                seed=1234)

    train_params["X_train"] = X_train
    train_params["y_train"] = y_train

    eval_params = {
        "X_test": X_test,
        "y_test": y_test
    }

    return train_params, eval_params

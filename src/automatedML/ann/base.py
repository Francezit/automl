from ..internal import get_training_metrics, TypeOfTask
from ..models import HyperparametersManager, get_global_hp_manager, get_layer_by_metadata
from ..models import BaseLayer

import numpy as np
import json
import os

DYNAMIC_ANN_STATE_BUILDING = 0
DYNAMIC_ANN_STATE_TRAINABLE = 1
DYNAMIC_ANN_STATE_TRAINED = 2


ANNMODEL_EXTENSION = ".aml"

class ANNModel():

    __input_size: tuple
    __output_size: tuple
    __type_of_task: TypeOfTask
    __seed: int

    __state: int
    __error: str
    __layer_info: list[BaseLayer]
    __keras_model: object

    def __init__(self, input_size: tuple, output_size: tuple, type_of_task: TypeOfTask | str, seed: int = None) -> None:

        self.__input_size = input_size
        self.__output_size = output_size
        self.__seed = seed if seed is not None else np.random.randint(0, 1000)
        if isinstance(type_of_task, str):
            self.__type_of_task = TypeOfTask(type_of_task)
        else:
            self.__type_of_task = type_of_task
        self.clear()

    @property
    def type_of_task(self):
        return self.__type_of_task

    @property
    def is_empty(self):
        return len(self.__layer_info) == 0

    @property
    def is_trainable(self):
        return self.__state == DYNAMIC_ANN_STATE_TRAINABLE

    @property
    def is_editable(self):
        return self.__state == DYNAMIC_ANN_STATE_BUILDING

    @property
    def is_ready(self):
        return self.__state == DYNAMIC_ANN_STATE_TRAINED

    @property
    def state(self):
        return self.__state

    @property
    def number_hidden_layers(self) -> int:
        assert self.is_trainable or self.is_ready
        return len(self.__keras_model.layers)

    @property
    def number_parameters(self) -> int:
        assert self.is_trainable or self.is_ready
        return self.__keras_model.count_params()

    @property
    def freeze_error(self):
        return self.__error

    def clear(self):
        self.__keras_model = None
        self.__state = DYNAMIC_ANN_STATE_BUILDING
        self.__error = None
        self.__layer_info = []

    def get_layer_metadatas(self) -> list:
        raise Exception("Not implemented")

    def get_info_dict(self) -> dict:
        return {
            "layers": self.get_layer_metadatas(),
            "number_parameters": self.number_parameters,
            "number_hidden_layers": self.number_hidden_layers,
            "type_of_task": self.__type_of_task.value,
            "seed": self.__seed,
            "input_size": self.__input_size,
            "output_size": self.__output_size,
            "state": self.__state
        }

    def get_layers(self) -> list:
        return [l.clone() for l in self.__layer_info]

    def get_layer_keys(self) -> list:
        return [l.uniqueKey for l in self.__layer_info]

    def get_layer_metadatas(self) -> list:
        return [l.metadata for l in self.__layer_info]

    def resize_layer(self, layerIndex: int):
        assert self.is_editable
        del self.__layer_info[layerIndex+1:]

    def add_layers(self, layers: list[BaseLayer]):
        assert self.is_editable

        if not isinstance(layers, list):
            layers = [layers]

        for layer in layers:
            self.__layer_info.append(layer.clone())

    def remove_layer(self, layerIndex: int):
        assert self.is_editable
        del self.__layer_info[layerIndex]

    def edit(self):
        if not self.is_editable:
            self.__keras_model = None
            self.__state = DYNAMIC_ANN_STATE_BUILDING

    def model_plot(self, filename: str, show_shapes=True):
        assert self.is_trainable or self.is_ready
        from ..engine import plot_model

        plot_model(
            self.__keras_model,
            to_file=filename,
            show_shapes=show_shapes,
            show_dtype=False,
            show_layer_names=False,
            rankdir="TB",
            expand_nested=True,
            dpi=100,
            layer_range=None,
            show_layer_activations=True
        )

    def model_summary(self, print_fn=None, filename: str = None):
        assert self.is_trainable or self.is_ready

        if filename is not None:
            if os.path.exists(filename):
                os.remove(filename)

            def myprint(s):
                with open(filename, 'a') as f:
                    print(s, file=f)
            print_fn = myprint

        if print_fn is None:
            print_fn = print

        self.__keras_model.summary(print_fn=print_fn)

    def freeze(self) -> bool:
        assert self.is_editable
        from ..engine import layers, initializers, Model

        # check if there is at least one layer
        if len(self.__layer_info) == 0:
            self.__error = "There is no layers"
            return False

        # check if the network has trainable parameters
        if not any([item.is_trainable() for item in self.__layer_info]):
            self.__error = "Network has not trainable parameters"
            return False

        # check if two layers with the same class and not trainable are linked together
        prev_layer: BaseLayer = self.__layer_info[0]
        for i in range(1, len(self.__layer_info)):
            curr_layer: BaseLayer = self.__layer_info[i]
            if not curr_layer.is_trainable() and prev_layer.get_type() == curr_layer.get_type():
                self.__error = "Two layers with the same class and not trainable are linked together"
                return False
            prev_layer = curr_layer
        del prev_layer

        try:

            initializer = initializers.GlorotUniform(seed=self.__seed)

            # input layer
            inputs = layers.Input(shape=self.__input_size)

            # hidden layer
            x = inputs
            for layer in self.__layer_info:
                x = layer.compile(x, initializer=initializer)

            # add output layers
            output = None
            if self.__type_of_task.is_forecasting:
                output = layers.Dense(
                    units=self.__output_size[1],
                    kernel_initializer=initializer
                )(x)
            elif self.__type_of_task.is_binary_task:
                x = layers.Flatten()(x)
                output = layers.Dense(
                    units=1,
                    kernel_initializer=initializer,
                    activation='sigmoid'
                )(x)
            elif self.type_of_task.is_discrete_task:
                x = layers.Flatten()(x)
                output = layers.Dense(
                    units=self.__output_size,
                    kernel_initializer=initializer,
                    activation='softmax'
                )(x)
            elif self.__type_of_task.is_continue_task:
                x = layers.Flatten()(x)
                if len(self.__output_size) > 1:
                    x = layers.Dense(
                        units=np.prod(self.__output_size),
                        kernel_initializer=initializer
                    )(x)
                    output = layers.Reshape(self.__output_size)(x)
                else:
                    output = layers.Dense(
                        units=self.__output_size,
                        kernel_initializer=initializer
                    )(x)
            else:
                raise Exception(
                    "Impossible to determinate the output layer, task not valid")

            self.__keras_model = Model(inputs, output)
            self.__error = None
            self.__state = DYNAMIC_ANN_STATE_TRAINABLE
            return True
        except Exception as err:
            self.__keras_model = None
            self.__error = str(err)
            return False

    def make(self, model: list[BaseLayer]) -> bool:
        assert self.is_empty and self.is_editable
        self.add_layers(model)
        return self.freeze()

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_validation: np.ndarray = None,
              y_validation: np.ndarray = None,
              epochs: int = 2,
              shuffle: bool = True,
              batch_size: int = None,
              optimizer: str = "adam",
              optimizer_configs: dict = None,
              num_workers: int = None,  # TODO NOT SUPPORTET YET
              loss_function_name: str = None,
              use_early_stopping: bool = False,
              use_tensor_board: str = None,
              use_kfold: int = 1,
              extract_history: bool = False):
        assert self.is_trainable

        from ..engine import training_optimizers, training_callbacks, Model, train_model, train_model_with_kfold

        # get loss
        loss_type, metrics_type = get_training_metrics(self.__type_of_task)
        if loss_function_name is not None:
            loss_type = loss_function_name

        # get optimizer
        training_optimizer = optimizer
        if optimizer_configs is not None:
            training_optimizer = training_optimizers.get({
                "class_name": optimizer,
                "config": optimizer_configs
            })

        # set callbacks
        callbacks = []
        if use_early_stopping:
            callbacks.append(training_callbacks.EarlyStopping(
                monitor='loss',
                patience=1,
                verbose=0,
                restore_best_weights='True',
                min_delta=0.1
            ))
        if use_tensor_board:
            callbacks.append(training_callbacks.TensorBoard(use_tensor_board))

        # set validation data
        validation_data: tuple = None
        if X_validation is not None and y_validation is not None:
            validation_data = (X_validation, y_validation)

        model: Model
        history: list
        if use_kfold and use_kfold > 1:
            model, history = train_model_with_kfold(
                model=self.__keras_model,
                X_train=X_train,
                y_train=y_train,
                loss_type=loss_type,
                training_optimizer=training_optimizer,
                metrics_type=metrics_type,
                epochs=epochs,
                shuffle=shuffle,
                validation_data=validation_data,
                batch_size=batch_size,
                callbacks=callbacks
            )

        else:
            model, history = train_model(
                model=self.__keras_model,
                X_train=X_train,
                y_train=y_train,
                loss_type=loss_type,
                training_optimizer=training_optimizer,
                metrics_type=metrics_type,
                epochs=epochs,
                shuffle=shuffle,
                validation_data=validation_data,
                batch_size=batch_size,
                callbacks=callbacks
            )

        # get loss
        loss = history.history['loss'][-1]
        if np.isnan(loss):
            loss = np.inf

        other_metrics = [history.history[x][-1]
                         for x in metrics_type if x in history.history]

        self.__state = DYNAMIC_ANN_STATE_TRAINED
        if extract_history:
            return [
                x
                for x in history.history['loss']
            ], [
                history.history[x]
                for x in metrics_type
                if x in history.history
            ]
        else:
            return loss, other_metrics

    def eval(self, X_test: np.ndarray, y_test: np.ndarray):
        assert self.is_ready

        val = self.__keras_model.evaluate(X_test, y_test, verbose=False)
        loss: float = float(val[0])
        other_metrics: list[float] = list(val[1:])
        return loss, other_metrics

    def predict(self, X: np.ndarray):
        assert self.is_ready
        y_predicted = np.array(self.__keras_model(X))
        return y_predicted

    def save(self, filename: str):
        assert self.is_trainable or self.is_ready
        from ..engine import save_model, create_zip, create_temp_context

        if not filename.endswith(ANNMODEL_EXTENSION):
            filename = f"{filename}{ANNMODEL_EXTENSION}"

        with create_temp_context(os.path.dirname(filename)) as tempcontext:
            foldername = tempcontext.foldername

            # set filenames
            modelfilename = os.path.join(
                foldername,
                "model.keras"
            )
            metadataFilename = os.path.join(
                foldername,
                "metadata.json"
            )
            summaryFilename = os.path.join(
                foldername,
                "summary.txt"
            )

            # save contets
            save_model(self.__keras_model, modelfilename)

            with open(metadataFilename, "w") as fp:
                json.dump(self.get_info_dict(), fp)

            if summaryFilename:
                self.model_summary(filename=summaryFilename)

            create_zip(filename, [
                modelfilename,
                metadataFilename,
                summaryFilename
            ])

    @staticmethod
    def load_metadata(filename: str) -> dict:
        if not filename.endswith(ANNMODEL_EXTENSION):
            raise Exception("File not supported")
        assert os.path.exists(filename), "File does not"

        from ..engine import extract_zip, create_temp_context

        with create_temp_context(os.path.dirname(filename)) as tempcontext:
            foldername = tempcontext.foldername

            metadataFilename = os.path.join(foldername, "metadata.json")

            # restore data
            extract_zip(filename, foldername)

            with open(metadataFilename, "r") as fp:
                info_dict = json.load(fp)
            return info_dict

    @staticmethod
    def load(filename: str, hp_manager: HyperparametersManager = None, ignore_metadata: bool = False, rebuild_model: bool = False) -> "ANNModel":
        # TODO correggere ignore metadata specialemente quando c'e una modalità con una sequenza
        if not filename.endswith(ANNMODEL_EXTENSION):
            raise Exception("File not supported")

        assert os.path.exists(filename), "File does not"

        if hp_manager is None:
            hp_manager = get_global_hp_manager()

        from ..engine import load_model, Model, extract_zip, create_temp_context

        keras_model: Model
        info_dict: dict
        with create_temp_context(os.path.dirname(filename)) as tempcontext:
            foldername = tempcontext.foldername

            # set filenames
            modelfilename = os.path.join(foldername, "model.keras")
            metadataFilename = os.path.join(foldername, "metadata.json")

            # restore data
            extract_zip(filename, foldername)

            # load model
            if not rebuild_model:
                keras_model = load_model(modelfilename)
            with open(metadataFilename, "r") as fp:
                info_dict = json.load(fp)
            if rebuild_model:
                info_dict["state"] = DYNAMIC_ANN_STATE_BUILDING
                keras_model = None

        # init model
        annmodel = ANNModel(input_size=info_dict["input_size"],
                            output_size=info_dict["output_size"],
                            seed=info_dict["seed"],
                            type_of_task=info_dict["type_of_task"])
        annmodel.__keras_model = keras_model
        annmodel.__state = info_dict["state"]
        if not ignore_metadata:
            annmodel.__layer_info = [
                get_layer_by_metadata(
                    metadata=x,
                    hp_manager=hp_manager
                )
                for x in info_dict["layers"]
            ]

        if rebuild_model and not ignore_metadata:
            if not annmodel.freeze():
                raise Exception(annmodel.freeze_error)
        return annmodel


__init__ = [
    "ANNModel", "create_ann_by_model",
    "DYNAMIC_ANN_STATE_BUILDING", "DYNAMIC_ANN_STATE_TRAINABLE", "DYNAMIC_ANN_STATE_TRAINED"
]

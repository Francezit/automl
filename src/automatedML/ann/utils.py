from ..models import BaseLayer
from ..datacontainer import DataContainer
from ..internal import TypeOfTask
from .architecture import ANNArchitecture, define_architecture_by_layers
from .base import ANNModel, HyperparametersManager


def create_ann_by_model(model: list[BaseLayer], raise_exception: bool = False, **ann_args):
    annmodel = ANNModel(**ann_args)
    valid: bool = annmodel.make(model)
    if valid:
        return annmodel
    elif raise_exception:
        raise Exception(annmodel.freeze_error)
    else:
        del annmodel
        return None


def create_ann_by_architecture(net: ANNArchitecture, **ann_args) -> ANNModel:
    return create_ann_by_model(net.to_model(), **ann_args)


def extract_architecture_from_ann(ann: ANNModel, name: str = None, hp_manager: HyperparametersManager = None):
    layers = ann.get_layers()
    net = define_architecture_by_layers(
        layers=layers,
        name=name,
        hp_manager=hp_manager
    )
    return net


class EvaluationArchitectureResult():
    __train_metric: tuple[float, list[float]]
    __eval_metric: tuple[float, list[float]]
    __annmodel: ANNModel

    def __init__(self, train_metric: tuple[float, list[float]], eval_metric: tuple[float, list[float]], annmodel: ANNModel) -> None:
        self.__train_metric = train_metric
        self.__eval_metric = eval_metric
        self.__annmodel = annmodel

    @property
    def train_metric(self):
        return self.__train_metric

    @property
    def eval_metric(self):
        return self.__eval_metric

    @property
    def annmodel(self):
        return self.__annmodel


def eval_architecture(net: ANNArchitecture, data_container: DataContainer,
                      seed: int = None, train_args: dict = {}) -> EvaluationArchitectureResult:

    # create model
    annmodel = create_ann_by_model(
        model=net.to_model(),
        raise_exception=True,
        input_size=data_container.input_size,
        output_size=data_container.output_size,
        type_of_task=data_container.type_of_task,
        seed=seed
    )

    # set default model
    base_train_args = {
        "batch_size": 64,
        "epochs": 1,
        "use_early_stopping": True
    }
    base_train_args.update(train_args)

    # train
    X_train, y_train = data_container.get_trainingset()
    train_loss = annmodel.train(
        X_train=X_train,
        y_train=y_train,
        **base_train_args
    )

    # eval
    X_test, y_test = data_container.get_testset()
    eval_loss = annmodel.eval(
        X_test=X_test,
        y_test=y_test
    )

    return EvaluationArchitectureResult(
        eval_metric=eval_loss,
        train_metric=train_loss,
        annmodel=annmodel
    )


def plot_architecture(net: ANNArchitecture, filename: str):
    model = create_ann_by_architecture(
        net=net,
        input_size=(100, 1),
        output_size=1,
        type_of_task=TypeOfTask.BINARY_CLASSIFICATION
    )
    model.model_plot(filename, show_shapes=False)
    pass


def check_layer_sequence(model: list[BaseLayer], **ann_args):
    annmodel = ANNModel(**ann_args)
    valid: bool = annmodel.make(model)
    status: str = annmodel.freeze_error if not valid else None
    del annmodel
    return valid, status


__all__ = [
    "create_ann_by_model", "create_ann_by_architecture", "extract_architecture_from_ann",
    "eval_architecture", "check_layer_sequence", "plot_architecture"
]

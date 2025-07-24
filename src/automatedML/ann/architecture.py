import json
from ..models import parse_layers, parse_layer
from ..models import BaseLayer, HyperparametersManager, HyperparameterFormat, get_global_hp_manager


class ANNArchitecture():
    __layers: list[BaseLayer]
    __train_args: dict  # TODO to include

    def __init__(self, hp_manager: HyperparametersManager = None, **kargs):

        if not kargs.pop("is_cloned", False):

            layers = kargs.pop("layers", None)
            if layers is None:
                if hp_manager is None:
                    hp_manager = get_global_hp_manager()
                layers = self._get_layer_model(hp_manager)

            if not isinstance(layers, list) or len(layers) == 0:
                raise Exception("Layers not supported")

            train_args = kargs.pop("train_args", None)
            if train_args is None:
                if hp_manager is None:
                    hp_manager = get_global_hp_manager()
                train_args = self._get_train_args(hp_manager)

            self.__layers = layers
            self.__train_args = train_args

    @property
    def name(self) -> str:
        raise Exception("Not implemented")

    @property
    def count_hyperparameters(self):
        return len(self.get_hp_vector())
    
    
    @property
    def count_layers(self):
        return len(self.__layers)

    def _get_train_args(self, hp_manager: HyperparametersManager) -> dict:
        return {}

    def _get_layer_model(self, hp_manager: HyperparametersManager) -> list[BaseLayer]:
        raise Exception("Not implemented")

    def to_model(self) -> list[BaseLayer]:
        return self.__layers
    
    def to_sequence(self):
        return [x.layerKey for x in self.__layers]

    def get_train_args(self):
        return self.__train_args

    def clone(self):
        net = ANNArchitecture(is_cloned=True)
        net.__layers = [x.clone() for x in self.__layers]
        net.__train_args = self.__train_args.copy()
        return net

    def get_hp_vector(self) -> list:
        hp_vector = []
        for v in self.__layers:
            for h in v.get_hpvector():
                hp_vector.append(h)
        return hp_vector

    def set_hp_vector(self, hp_vector: list, ignore_constraint: bool = False):
        i = 0
        for l in self.__layers:
            n = l.count_hyperparameters
            if n > 0:
                l.set_hpvector(hp_vector[i:i+n], ignore_constraint)
                i += n
        pass

    def get_hp_formats(self) -> list[HyperparameterFormat]:
        hp_vector = []
        for v in self.__layers:
            for h in v.get_hyperparameter_formats():
                hp_vector.append(h)
        return hp_vector

    def get_hyperparameter_dic(self):
        hpdict = {}
        for i, layer in enumerate(self.__layers):
            for n, v in layer.get_hyperparameter_dic().items():
                hpdict[f"L{i}_{n}"] = v
        return hpdict


class WrapperANNArchitecture(ANNArchitecture):
    __name: str

    def __init__(self, hp_manager: HyperparametersManager = None, **kargs):
        assert "layers" in kargs, "Layers have not been declared"

        super().__init__(hp_manager, **kargs)
        self.__name = kargs.get("name", None)
        if self.__name is None:
            self.__name = "CustomANN"

    @property
    def name(self) -> str:
        return self.__name


def define_architecture_by_layers(layers: list, name: str = None, hp_manager: HyperparametersManager = None, train_args: dict = None) -> ANNArchitecture:
    if hp_manager is None:
        hp_manager = get_global_hp_manager()

    if not isinstance(layers, list):
        layers = [layers]

    return WrapperANNArchitecture(
        hp_manager=hp_manager,
        name=name,
        layers=parse_layers(
            items=layers,
            hp_manager=hp_manager
        ),
        train_args=train_args
    )


def save_architecture(filename: str, net: ANNArchitecture):
    layers_keys = []
    layer_hp_formats = []
    for layer in net.to_model():
        layers_keys.append(layer.layerKey)
        layer_hp_formats.append([
            x.to_dict()
            for x in layer.get_hyperparameter_formats()
        ])

    with open(filename, "w") as fp:
        json.dump({
            "layers_keys": layers_keys,
            "name": net.name,
            "layer_hp_formats": layer_hp_formats,
            "train_args": net.get_train_args()
        }, fp)


def load_architecture(filename: str, hp_manager: HyperparametersManager = None) -> ANNArchitecture:

    if hp_manager is None:
        hp_manager = get_global_hp_manager()

    with open(filename, "r") as fp:
        data: dict = json.load(fp)

    layers_keys = data["layers_keys"]
    name = data["name"]
    train_args = data.get("train_args", None)

    hp_mangers = [
        hp_manager.update([
            HyperparameterFormat(**y)
            for y in x
        ])
        for x in data["layer_hp_formats"]
    ]

    layers = [
        parse_layer(x, hp)
        for x, hp in zip(layers_keys, hp_mangers)
    ]

    return define_architecture_by_layers(
        layers=layers,
        hp_manager=hp_manager,
        name=name,
        train_args=train_args
    )


__all__ = ["ANNArchitecture", "define_architecture_by_layers",
           "save_architecture", "load_architecture"]

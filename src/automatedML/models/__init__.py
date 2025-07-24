from .baselayer import *
from .wrapperlayer import *
from .hpmanager import *
from .layermap import *


from .activationlayer import ActivationLayer
from .dropoutlayer import DropoutLayer
from .fullyconnectedlayer import FullyConnectedLayer
from .fullyconnected_with_activation_layer import FullyConnectedWithActivationLayer
from .fullyconnectedblock import FullyConnectedBlock
from .lstmlayer import LSTMLayer
from .grulayer import GRULayer
from .bilstmllayer import BiLSTMLayer
from .recurrentlayer import RecurrentLayer
from .attentionrecurrentlayer import AttentionRecurrentLayer
from .fullyconnectedRNNlayer import FullyConnectedRNNLayer
from .convnextblock import ConvNeXtBlock
from .convolutionallayer import ConvolutionalLayer
from .convolutional1dlayer import Convolutional1DLayer
from .convolutional2dlayer import Convolutional2DLayer
from .deconvolutionallayer import DeconvolutionalLayer
from .deconvolutional1dlayer import Deconvolutional1DLayer
from .deconvolutional2dlayer import Deconvolutional2DLayer
from .convolutionalblock import ConvolutionalBlock
from .maxpoolinglayer import MaxPoolingLayer
from .maxpooling1dlayer import MaxPooling1DLayer
from .maxpooling2dlayer import MaxPooling2DLayer
from .averagepoolinglayer import AveragePoolingLayer
from .averagepooling1dlayer import AveragePooling1DLayer
from .averagepooling2dlayer import AveragePooling2DLayer
from .poolinglayer import PoolingLayer
from .pooling1dlayer import Pooling1DLayer
from .pooling2dlayer import Pooling2DLayer
from .upsampling1d import UpSampling1DLayer
from .upsampling2d import UpSampling2DLayer
from .upsampling import UpSamplingLayer
from .batchnormalization import BatchNormalizationLayer
from .localresponsenormalization import LocalResponseNormalizationLayer
from .flattenlayer import FlattenLayer
from .convolutionalLSTMlayer import ConvolutionalLSTMLayer
from .residualblock import ResidualBlock
from .autoencoder1dblock import AutoEncoder1DBlock
from .autoencoder2dblock import AutoEncoder2DBlock
from .algebralayer import AlgebraOperatorLayer
from .identitylayer import IdentityLayer
from .filterinputlayer import FilterInputLayer
from .inceptionmodule import InceptionModule
from .resizinglayer import ResizingLayer

# ------------------------------------------------
__custom_folder = "./custom-layers"
__layer_classes: list = [
    FullyConnectedLayer,
    FullyConnectedWithActivationLayer,
    FullyConnectedBlock,
    ActivationLayer,
    DropoutLayer,
    FlattenLayer,
    BatchNormalizationLayer,
    LocalResponseNormalizationLayer,
    LSTMLayer,
    GRULayer,
    BiLSTMLayer,
    FullyConnectedRNNLayer,
    RecurrentLayer,
    ConvolutionalLSTMLayer,
    ConvolutionalLayer,
    Convolutional1DLayer,
    Convolutional2DLayer,
    ConvolutionalBlock,
    DeconvolutionalLayer,
    Deconvolutional1DLayer,
    Deconvolutional2DLayer,
    MaxPoolingLayer,
    MaxPooling1DLayer,
    MaxPooling2DLayer,
    AveragePoolingLayer,
    AveragePooling1DLayer,
    AveragePooling2DLayer,
    PoolingLayer,
    Pooling1DLayer,
    Pooling2DLayer,
    AlgebraOperatorLayer,
    UpSampling1DLayer,  # TOTEST
    UpSampling2DLayer,  # TOTEST
    UpSamplingLayer,  # TOTEST
    ResidualBlock,
    AutoEncoder1DBlock,  # TOTEST
    AutoEncoder2DBlock,  # TOTEST
    # Transformer, #TODO
    ConvNeXtBlock,
    IdentityLayer,
    InceptionModule  # TOTEST
]
__layer_classes_map = dict(
    [(x().get_type().lower(), x) for x in __layer_classes]
)
__layer_custom_names = []


def get_all_class_names():
    return list(__layer_classes_map.keys())


def load_custom_layers(filenames: list, handle_exception: bool = False):

    if isinstance(filenames, str):
        filenames = [filenames]

    class_imported = []

    # compute context
    global_context = globals()
    context = {
        x: global_context[x]
        for x in global_context
        if not x.startswith("_") and x[0].isupper()
    }

    for filename in filenames:
        try:
            print(f"LoadCustomLayer: start importing {filename}")

            # read the file
            script_text: str
            with open(filename, "r") as fp:
                script_text = fp.read()

            # run scripts
            data = {}
            exec(script_text, context, data)
            del script_text

            # finds the layer classes
            layer_class_names = data.get("__all__", [])
            if len(layer_class_names) == 0:
                print(f"No layer classes was found in {filename}")
            else:
                for layer_class_name in layer_class_names:
                    # instance the class
                    layer_class = data.get(layer_class_name, None)
                    if layer_class is None:
                        raise Exception(
                            f"Layer class {layer_class_name} not found in {filename}")

                    # check class
                    layer = layer_class()
                    if isinstance(layer, BaseLayer):
                        type = layer.get_type().lower()
                        if type in __layer_classes_map:
                            print(
                                f"{type} has already been registered in {filename}")

                        __layer_classes.append(layer_class)
                        __layer_classes_map[type] = layer_class
                        __layer_custom_names.append(type)
                        class_imported.append((type, layer_class))
                        print(f"{type} imported")
                    else:
                        print(f"{layer_class_name} not supported")

        except Exception as err:
            if handle_exception:
                print(f"LoadCustomLayer: {err}")
            else:
                raise
    return class_imported


def get_custom_layers():
    return [(x, __layer_classes_map[x]) for x in __layer_custom_names]


def remove_custom_layers(class_names: list = None):
    if class_names is None:
        class_names = __layer_custom_names.copy()

    for class_name in class_names:
        class_name = class_name.lower()
        __layer_custom_names.remove(class_name)
        item = __layer_classes_map.pop(class_names)
        __layer_classes.remove(item)


def __load_internal_custom_layers():
    import os
    if os.path.exists(__custom_folder):
        filenames = [
            element for element in os.listdir(__custom_folder)
            if element.endswith(".py")
        ]
        load_custom_layers(filenames, True)


__load_internal_custom_layers()


# ------------------------------------------------

def get_layers_by_class_name(class_name: str, hp_manager: HyperparametersManager, useMetadata: bool = False) -> list[BaseLayer | list]:

    from itertools import product

    default_layer: BaseLayer = get_layer_by_key(class_name, hp_manager)

    hp_lists = default_layer.get_hyperparameter_bounds()
    if len(hp_lists) > 0:

        hp_map = list(product(*hp_lists))

        layers = []
        for hp_vector in hp_map:
            if useMetadata:
                layers.append([class_name]+list(hp_vector))
            else:
                layer: BaseLayer = default_layer.clone()
                if layer.has_hyperparameters:
                    layer.set_hpvector(hp_vector)
                layers.append(layer)
        return layers
    else:
        if useMetadata:
            return [default_layer.metadata]
        else:
            return [default_layer]


def get_layer_by_key(layer_key: str, hp_manager: HyperparametersManager) -> BaseLayer:

    # extract info
    key_parts = layer_key.split('-')
    type_name = key_parts[0]

    # extract layer params
    layers_params: dict[str, object] = {}
    for i in range(1, len(key_parts)):
        code = key_parts[i].split('=')
        layers_params[code[0]] = code[1]

    # check parallel and sequential branches
    base_params, type_name = _extract_modes_from_type(type_name)
    parallel_class = type_name.split('/')
    sequential_class = type_name.split('+')
    if len(parallel_class) > 1:
        branch_layers = []
        for branch_index, branch_class in enumerate(parallel_class):
            branch_param_code = f"b{branch_index}"

            branch_key = [branch_class]
            for x in list(layers_params.keys()):
                if x.startswith(branch_param_code):
                    branch_key.append(
                        f"{x.replace(branch_param_code,'',1)}={layers_params.pop(x)}"
                    )
            branch_key = '-'.join(branch_key)

            l = get_layer_by_key(branch_key, hp_manager)
            branch_layers.append(l)

        layer: BaseLayer = build_parallel_block(
            layers=branch_layers,
            hp_manager=hp_manager,
            **base_params
        )
        layer.update_hyperparameters(**layers_params)

        return layer
    elif len(sequential_class) > 1:
        sub_layers = []
        for seq_index, seq_class in enumerate(sequential_class):
            seq_param_code = f"l{seq_index}"

            seq_key = [seq_class]
            for x in list(layers_params.keys()):
                if x.startswith(seq_param_code):
                    seq_key.append(
                        f"{x.replace(seq_param_code,'',1)}={layers_params.pop(x)}"
                    )
            seq_key = '-'.join(seq_key)

            l = get_layer_by_key(seq_key, hp_manager)
            sub_layers.append(l)

        layer: BaseLayer = build_sequential_block(
            layers=sub_layers,
            hp_manager=hp_manager,
            **base_params
        )
        layer.update_hyperparameters(**layers_params)

        return layer
    else:
        type_parts = type_name.split(":")
        className = type_parts[0]
        # base_params, className = _extract_modes_from_type(type_parts[0])

        # extract static params
        for i in range(1, len(type_parts)):
            code = type_parts[i].split('=')
            base_params[code[0]] = code[1]

        # create layer
        layer: BaseLayer = __layer_classes_map[className.lower()](
            hp_manager=hp_manager,
            **base_params
        )
        layer.update_hyperparameters(**layers_params)

        return layer


def parse_layer(item: str | BaseLayer | list | dict | type, hp_manager: HyperparametersManager) -> BaseLayer:
    if isinstance(item, str):
        return get_layer_by_key(item, hp_manager)
    elif isinstance(item, BaseLayer):
        return item
    elif isinstance(item, list):
        sub_layers = parse_layers(item, hp_manager)
        if len(sub_layers) > 1:
            return build_sequential_block(layers=sub_layers)
        elif len(sub_layers) == 1:
            return sub_layers[0]
        else:
            raise Exception("Sequential List not valid")
    elif isinstance(item, dict):
        sub_layers = []
        params = {}
        for obj_key in item:
            obj_value = item[obj_key]
            if isinstance(obj_value, list):
                l = parse_layer(obj_value, hp_manager)
                sub_layers.append(l)
            else:
                params[obj_key] = obj_value

        if len(sub_layers) > 1:
            return build_parallel_block(layers=sub_layers, **params)
        elif len(sub_layers) == 1:
            return sub_layers[0]
        else:
            raise Exception("Parallel List not valid")
    elif isinstance(item, type):
        return item(hp_manager=hp_manager)
    else:
        raise Exception("Not supported")


def parse_layers(items: list[str | BaseLayer | list | dict], hp_manager: HyperparametersManager) -> list[BaseLayer]:
    layers = []
    for item in items:
        layers.append(parse_layer(item, hp_manager))
    return layers


def get_layer_by_metadata(metadata: list, hp_manager: HyperparametersManager) -> BaseLayer:
    className: str = metadata[0]
    hps: list = metadata[1:]

    layer: BaseLayer = get_layer_by_key(className, hp_manager)
    if layer.has_hyperparameters:
        layer.set_hpvector(hpvector=hps,
                           ignore_constraint=True)
    return layer


def compute_layermap_new(hp_manager: HyperparametersManager, class_names: list = None) -> LayerMap:
    if class_names is None:
        class_names = get_all_class_names()

    layer_map = dict()
    for class_name in class_names:
        default_layer: BaseLayer = get_layer_by_key(class_name, hp_manager)
        hp_lists = default_layer.get_hyperparameter_bounds()

        layer_map[class_name] = hp_lists

    return LayerMap(layer_map)


def compute_layermap(hp_manager: HyperparametersManager, class_names: list = None) -> dict[str | list]:
    if class_names is None:
        class_names = get_all_class_names()

    layer_map = dict()
    for class_name in class_names:
        layer_map[class_name] = get_layers_by_class_name(
            class_name=class_name,
            hp_manager=hp_manager,
            useMetadata=True
        )
    return layer_map


def get_statistic_from_layermap(layer_map: dict, hp_manager: HyperparametersManager):
    import pandas as pd

    stat = pd.DataFrame(columns=[
        "class",
        "hyperparams_number",
        "hyperparams_names",
        "hyperparams_value",
        "instances"
    ])

    for class_name in layer_map:
        class_value = layer_map[class_name]

        layer: BaseLayer = get_layer_by_key(class_name, hp_manager)
        class_hps_name = layer.get_hyperparameter_names()
        class_hps_values = [
            layer.get_hyperparameter_format(name).size
            for name in class_hps_name
        ]

        stat.loc[len(stat)] = {
            "class": class_name,
            "hyperparams_number": len(class_hps_name),
            "hyperparams_names": ','.join(class_hps_name),
            "hyperparams_value": ','.join([str(x) for x in class_hps_values]),
            "instances": len(class_value)
        }
    return stat

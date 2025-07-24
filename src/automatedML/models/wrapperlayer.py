from .baselayer import BaseSequentialBlock, BaseParallelBlock, BaseLayer
from .identitylayer import IdentityLayer
from .hpmanager import HyperparameterFormat, HyperparametersManager


class WrapperSequentialBlock(BaseSequentialBlock):

    def get_type(self) -> str:
        st = self.get_type_sublayers()
        return f"({'+'.join(st)})"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class WrapperParallellBlock(BaseParallelBlock):

    def get_type(self) -> str:
        st = self.get_type_sublayers()
        return f"({'/'.join(st)})"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LayerCompileFunction():
    def __call__(self, prev_layer, initializer, hps: dict[str, object]):
        raise Exception("Not implemented")
    
class LambdaLayerFunction():
    def __call__(self, x):
        raise Exception("Not implemented")

class WrapperBaseLayer(BaseLayer):

    _wrapper_compile: LayerCompileFunction
    _wrapper_type: str
    _wrapper_trainable: bool
    _wrapper_hp: dict[str, HyperparameterFormat]
    _is_lambda: bool

    def __init__(self, wrapper_compile: LayerCompileFunction|LambdaLayerFunction, wrapper_type: str, is_lambda:bool,
                 wrapper_trainable: bool, wrapper_hp: dict[str, HyperparameterFormat], **kwargs) -> None:

        self._wrapper_compile = wrapper_compile
        self._wrapper_type = wrapper_type
        self._wrapper_trainable = wrapper_trainable
        self._wrapper_hp = wrapper_hp
        self._is_lambda = is_lambda

        super().__init__( disable_clone=True,**kwargs)

    def is_trainable(self) -> bool:
        return self._wrapper_trainable

    def get_type(self) -> str:
        return self._wrapper_type

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        for hp_name, hp_format in self._wrapper_hp:
            self._register_hyperparameter(
                name=hp_name,
                format=hp_format
            )

    def _internal_compile(self, prev_layer, initializer):

        if self._is_lambda:
            from .engine import layers
            return layers.Lambda(self._wrapper_compile)(prev_layer)
        else:
            hps = self.get_hyperparameter_dic()
            return self._wrapper_compile(prev_layer, initializer, hps)


def build_custom_layer(fn_compile: LayerCompileFunction, type: str, trainable: bool = False,
                       hyperparamenter_config: dict[str, HyperparameterFormat] = {}, **kargs):

    return WrapperBaseLayer(
        wrapper_compile=fn_compile,
        wrapper_type=type,
        wrapper_trainable=trainable,
        wrapper_hp=hyperparamenter_config,
        is_lambda=False,
        **kargs
    )

def build_lambda_layer(fn: LambdaLayerFunction, **kargs):
   
    return WrapperBaseLayer(
        wrapper_compile=fn,
        wrapper_type="Lambda",
        wrapper_trainable=False,
        wrapper_hp={},
        is_lambda=True,
        **kargs
    )


def build_sequential_block(layers: BaseLayer | list[BaseLayer], **kargs) -> BaseSequentialBlock:
    if isinstance(layers, BaseLayer):
        return layers
    elif isinstance(layers, list):
        assert len(layers) > 0, "The number of layers must be greater than 0"
        if len(layers) == 1:
            return layers[0]
        elif len(layers) > 1:
            return WrapperSequentialBlock(layers=layers, **kargs)


def build_parallel_block(layers: list[BaseLayer],  **kargs) -> BaseParallelBlock:
    assert len(layers) > 1, "The number of layers must be greater than 1"
    return WrapperParallellBlock(layers=layers, **kargs)


def build_with_skip_connection(layers: BaseLayer | list[BaseLayer],  **kargs):
    kargs["skip_connection_mode"] = True
    kargs["skip_connection"] = "concatenate"
    kargs["partial_freeze_mode"] = True
    return WrapperSequentialBlock(layers=layers, **kargs)


def build_cascade_block(layers: BaseLayer | list[BaseLayer], num_cascade_layer: int = None, **kargs):
    kargs["cascade_mode"] = True
    kargs["partial_freeze_mode"] = True
    if num_cascade_layer is not None:
        kargs["num_cascade_layer"] = num_cascade_layer
    return WrapperSequentialBlock(layers=layers, **kargs)


def transform_sequential_to_parallel(layer: BaseLayer, n_parallel: int, **kargs):
    assert n_parallel > 1

    layers = []
    for _ in range(n_parallel):
        layers.append(layer.clone())
    return WrapperParallellBlock(layers=layers, **kargs)


__all__ = ["build_sequential_block", "build_parallel_block", "build_with_skip_connection","build_cascade_block",
           "transform_sequential_to_parallel", "build_custom_layer", "LayerCompileFunction","build_lambda_layer","LambdaLayerFunction"]

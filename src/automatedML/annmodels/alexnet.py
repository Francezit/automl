from ..models import *
from ..ann import ANNArchitecture


class AlexNetArchitecture(ANNArchitecture):

    @property
    def name(self) -> str:
        return "AlexNet"

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def _register_hp_architecture(self, hp_manager: HyperparametersManager):
        pass

    def _get_layer_model(self, hp_manager: HyperparametersManager):
        return [  # 224,224,3
            ConvolutionalLayer(
                filters=64, kernel_size=11, strides=4,
                padding='valid',
                hp_manager=hp_manager
            ),
            PoolingLayer(
                type="max", pool_size=(3, 3), strides=(2, 2)
            ),
            ConvolutionalLayer(
                filters=192, kernel_size=(5, 5), padding='same',
                hp_manager=hp_manager
            ),
            PoolingLayer,
            ConvolutionalLayer,
            ConvolutionalLayer,
            ConvolutionalLayer,
            PoolingLayer,
            Flatten,
            FullyConnectedWithActivation,
            Dropout
        ]


def get_architecture():
    return AlexNetArchitecture

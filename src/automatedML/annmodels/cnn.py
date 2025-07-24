from ..models import *
from ..ann import ANNArchitecture


class CNNArchitecture(ANNArchitecture):

    @property
    def name(self) -> str:
        return "CNN"

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def _register_hp_architecture(self, hp_manager: HyperparametersManager):
        pass

    def _get_layer_model(self, hp_manager: HyperparametersManager):

        return [
            build_sequential_block(
                layers=[
                    BatchNormalizationLayer,
                    ConvolutionalLayer,
                    PoolingLayer
                ],
                hp_manager=hp_manager,
                cascade_mode=True,
                num_cascade_layer=1,
            ),
            FullyConnectedWithActivationLayer(
                size=50,
                hp_manager=hp_manager
            )
        ]


def get_architecture():
    return CNNArchitecture

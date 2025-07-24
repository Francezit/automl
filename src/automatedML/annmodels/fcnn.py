from ..models import *
from ..ann import ANNArchitecture


class FCNNArchitecture(ANNArchitecture):

    @property
    def name(self) -> str:
        return "FCNN"

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def _register_hp_architecture(self, hp_manager: HyperparametersManager):
        pass

    def _get_layer_model(self, hp_manager: HyperparametersManager):

        return [
            build_sequential_block(
                layers=[
                    FullyConnectedWithActivationLayer,
                    BatchNormalizationLayer,
                    DropoutLayer
                ],
                hp_manager=hp_manager,
                cascade_mode=True
            ),
            FullyConnectedWithActivationLayer(hp_manager=hp_manager)
        ]


def get_architecture():
    return FCNNArchitecture

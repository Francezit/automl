from ..models import *
from ..ann import ANNArchitecture


class MLPArchitecture(ANNArchitecture):

    @property
    def name(self) -> str:
        return "MLP"

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def _register_hp_architecture(self, hp_manager: HyperparametersManager):
        pass

    def _get_layer_model(self, hp_manager: HyperparametersManager):

        return [
            FullyConnectedBlock(
                cascade_mode=True,
                num_cascade_layer=3,
                hp_manager=hp_manager,
                l0_size=1000,
                l1_size=500,
                l2_size=250
            )
        ]


def get_architecture():
    return MLPArchitecture

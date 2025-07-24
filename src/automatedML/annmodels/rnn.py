from ..models import *
from ..ann import ANNArchitecture





class RNNArchitecture(ANNArchitecture):

    __recurrent_type: str

    @property
    def name(self) -> str:
        return "RNN"

    def __init__(self, recurrent_type: str = "LSTM", **kargs):
        self.__recurrent_type = recurrent_type
        super().__init__(**kargs)

    def _register_hp_architecture(self, hp_manager: HyperparametersManager):
        pass

    def _get_layer_model(self, hp_manager: HyperparametersManager):

        return [
            RecurrentLayer(
                unit=500,
                fun="tanh",
                type=self.__recurrent_type,
                hp_manager=hp_manager
            ),
            DropoutLayer(
                prob=0.2,
                hp_manager=hp_manager
            )
        ]


def get_architecture():
    return RNNArchitecture

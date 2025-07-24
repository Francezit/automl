from .baselayer import BaseSequentialBlock, HyperparametersManager
from .convolutional1dlayer import Convolutional1DLayer
from .dropoutlayer import DropoutLayer
from .deconvolutional1dlayer import Deconvolutional1DLayer
from .fullyconnectedlayer import FullyConnectedLayer

class AutoEncoder1DBlock(BaseSequentialBlock):

    def get_type(self):
        return "AutoEncoder1DBlock"

    def get_layers_in_block(self, hp_manager: HyperparametersManager):
        return [
            Convolutional1DLayer,
            DropoutLayer,
            Convolutional1DLayer,
            DropoutLayer,
            FullyConnectedLayer,
            Deconvolutional1DLayer,
            Deconvolutional1DLayer
        ]

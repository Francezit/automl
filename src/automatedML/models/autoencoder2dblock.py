from .baselayer import BaseSequentialBlock, HyperparametersManager
from .convolutional2dlayer import Convolutional2DLayer
from .dropoutlayer import DropoutLayer
from .deconvolutional2dlayer import Deconvolutional2DLayer
from .fullyconnectedlayer import FullyConnectedLayer


class AutoEncoder2DBlock(BaseSequentialBlock):

    def get_type(self):
        return "AutoEncoder2DBlock"

    def get_layers_in_block(self, hp_manager: HyperparametersManager):
        return [
            Convolutional2DLayer,
            DropoutLayer,
            Convolutional2DLayer,
            DropoutLayer,
            FullyConnectedLayer,
            Deconvolutional2DLayer,
            Deconvolutional2DLayer
        ]

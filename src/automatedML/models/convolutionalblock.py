from .baselayer import BaseSequentialBlock, HyperparametersManager
from .convolutionallayer import ConvolutionalLayer
from .poolinglayer import PoolingLayer
from .dropoutlayer import DropoutLayer
from .batchnormalization import BatchNormalizationLayer


class ConvolutionalBlock(BaseSequentialBlock):

    def get_type(self):
        return "ConvolutionalBlock"

    def get_layers_in_block(self, hp_manager: HyperparametersManager):
        layers = [
            ConvolutionalLayer,
            BatchNormalizationLayer,
            PoolingLayer
        ]
        return layers

from .baselayer import BaseSequentialBlock, HyperparametersManager
from .fullyconnected_with_activation_layer import FullyConnectedWithActivationLayer
from .dropoutlayer import DropoutLayer
from .batchnormalization import BatchNormalizationLayer


class FullyConnectedBlock(BaseSequentialBlock):

    def get_type(self):
        return "FullyConnectedBlock"

    def get_layers_in_block(self, hp_manager: HyperparametersManager):
        return [
            BatchNormalizationLayer,
            FullyConnectedWithActivationLayer,
            DropoutLayer
        ]

from .baselayer import BaseLayer, BaseSequentialBlock, HyperparametersManager
from .convolutional2dlayer import Convolutional2DLayer
from .batchnormalization import BatchNormalizationLayer
from .activationlayer import ActivationLayer
from .wrapperlayer import build_with_skip_connection


class ResidualBlock(BaseLayer):

    def get_type(self):
        return "ResidualBlock"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("filters",
                                      hp_manager.hp_residual_filters)
        self._register_hyperparameter("kernel",
                                      hp_manager.hp_residual_kernel)
        self._register_hyperparameter("fun",
                                      hp_manager.hp_residual_af_names)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers, pad, constant

        filters = self.get_hyperparameter("filters")
        kernel = self.get_hyperparameter("kernel")
        fun = self.get_hyperparameter("fun")

        fx = layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            padding='same',
        )(prev_layer)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Activation(activation=fun)(fx)

        fx = layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            padding='same'
        )(fx)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Activation(activation=fun)(fx)

        x_skip =  layers.Conv2D(
            filters=filters,
            kernel_size=(1,1),
            padding='same'
        )(prev_layer)
        x_skip = layers.BatchNormalization()(x_skip)

        out = layers.Add()([fx, x_skip])
        out = layers.ReLU()(out)
        return out

        # return advanced_layers.ResidualBlock(filters=filters,
        #                                    strides=strides,
        #                                     activation=fun,
        #                                     kernel_initializer=initializer,
        #                                     use_projection=True,
        #                                     resnetd_shortcut=resnetd_shortcut)(prev_layer)

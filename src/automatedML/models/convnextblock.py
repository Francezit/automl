from .baselayer import BaseLayer, HyperparametersManager

import numpy as np


class ConvNeXtBlock(BaseLayer):

    def get_type(self):
        return "ConvNeXtBlock"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("projection_dim",
                                      hp_manager.hp_convnext_projection_dim,
                                      default_value=4)
        self._register_hyperparameter("drop_path_rate",
                                      format=hp_manager.hp_convnext_drop_path_rate,
                                      default_value=0.0)

    def _internal_compile(self, inputs, initializer):
        from ..engine import layers, initializers, random, shape, floor, Variable, ones

        projection_dim = self.get_hyperparameter("projection_dim")
        drop_path_rate = self.get_hyperparameter("drop_path_rate")
        layer_scale_init_value = 1e-6

        gamma = Variable(
            initial_value=self.layer_scale_init_value * ones((projection_dim)),
            trainable=True,
            name='_gamma')

        x = inputs
        x = layers.DepthwiseConv2D(
            kernel_size=7,
            padding='same'
        )(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(4 * projection_dim, initializer=initializer)(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dense(projection_dim)(x)
        if gamma is not None:
            x = gamma * x

        return inputs + layers.DropPath(drop_path_rate)(x)


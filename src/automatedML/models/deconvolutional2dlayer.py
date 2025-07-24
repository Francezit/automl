from .baselayer import BaseLayer, HyperparametersManager


class Deconvolutional2DLayer(BaseLayer):

    def get_type(self):
        return "Deconvolutional2D"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("filters",
                                      hp_manager.hp_conv_filters)
        self._register_hyperparameter("kernel",
                                      hp_manager.hp_conv_kernel)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        filters = self.get_hyperparameter("filters")
        kernel = self.get_hyperparameter("kernel")

        if len(prev_layer.shape) == 4:
            x = prev_layer
        elif len(prev_layer.shape) == 3:
            x_shape = (prev_layer.shape[1], prev_layer.shape[2], 1)
            x = layers.Reshape(x_shape)(prev_layer)
        else:
            raise Exception("Not supported")

        x = layers.Conv2DTranspose(filters=filters,
                                   kernel_size=kernel,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=initializer)(x)
        return x

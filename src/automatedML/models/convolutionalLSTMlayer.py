from .baselayer import BaseLayer, HyperparametersManager


class ConvolutionalLSTMLayer(BaseLayer):

    def get_type(self):
        return "ConvolutionalLSTM"

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
            x = layers.ConvLSTM2D(filters=filters,
                                  kernel_size=kernel,
                                  activation='tanh',
                                  padding='same',
                                  kernel_initializer=initializer)(prev_layer)

        elif len(prev_layer.shape) == 3:
            x = layers.ConvLSTM1D(filters=filters,
                                  kernel_size=kernel,
                                  activation='tanh',
                                  padding='same',
                                  kernel_initializer=initializer)(prev_layer)

        else:
            x_shape = (prev_layer.shape[1], 1)
            x = layers.Reshape(x_shape)(prev_layer)

            x = layers.ConvLSTM1D(filters=filters,
                                  kernel_size=kernel,
                                  activation='tanh',
                                  padding='same',
                                  kernel_initializer=initializer)(x)
        return x

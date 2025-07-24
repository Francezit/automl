from .baselayer import BaseLayer, HyperparametersManager


class Pooling1DLayer(BaseLayer):

    def get_type(self):
        return "Pooling1D"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("pooling",
                                      hp_manager.hp_conv_pooling_type)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers, reduce_prod

        pooling_type = self.get_hyperparameter("pooling")

        if len(prev_layer.shape) > 3:
            x_shape = (int(reduce_prod(prev_layer.shape[1:-1])),
                       prev_layer.shape[-1])
            x = layers.Reshape(x_shape)(prev_layer)
        elif len(prev_layer.shape) == 3:
            x = prev_layer
        else:
            x_shape = (prev_layer.shape[1], 1)
            x = layers.Reshape(x_shape)(prev_layer)

        if pooling_type == "max":
            x = layers.MaxPooling1D()(x)
        elif pooling_type == "avg":
            x = layers.AveragePooling1D()(x)
        return x

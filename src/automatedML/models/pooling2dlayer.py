from .baselayer import BaseLayer, HyperparametersManager


class Pooling2DLayer(BaseLayer):

    def get_type(self):
        return "Pooling2D"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("pooling",
                                      hp_manager.hp_conv_pooling_type)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        pooling_type = self.get_hyperparameter("pooling")

        if len(prev_layer.shape) == 4:
            x = prev_layer
        elif len(prev_layer.shape) == 3:
            x_shape = (prev_layer.shape[1], prev_layer.shape[2], 1)
            x = layers.Reshape(x_shape)(prev_layer)
        else:
            raise Exception("Not supported")

        if pooling_type == "max":
            x = layers.MaxPooling2D()(x)
        elif pooling_type == "avg":
            x = layers.AveragePooling2D()(x)
        return x

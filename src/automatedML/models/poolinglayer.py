from .baselayer import BaseLayer, HyperparametersManager


class PoolingLayer(BaseLayer):

    def get_type(self):
        return "Pooling"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("pooling",
                                      hp_manager.hp_conv_pooling_type)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        pooling_type = self.get_hyperparameter("pooling")

        x = None
        if pooling_type == "max":

            if len(prev_layer.shape) == 4:
                x = layers.MaxPooling2D()(prev_layer)
            elif len(prev_layer.shape) == 3:
                x = layers.MaxPooling1D()(prev_layer)

        elif pooling_type == "avg":

            if len(prev_layer.shape) == 4:
                x = layers.AveragePooling2D()(prev_layer)
            elif len(prev_layer.shape) == 3:
                x = layers.AveragePooling1D()(prev_layer)

        if x is None:
            raise Exception("Not supported")
        return x

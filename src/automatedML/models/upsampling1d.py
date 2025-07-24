from .baselayer import BaseLayer, HyperparametersManager


class UpSampling1DLayer(BaseLayer):

    def get_type(self):
        return "UpSampling1D"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("size",
                                      hp_manager.hp_upsampling_size)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers, reduce_prod

        size = self.get_hyperparameter("size")

        if len(prev_layer.shape) > 3:
            x_shape = (int(reduce_prod(prev_layer.shape[1:-1])),
                       prev_layer.shape[-1])
            x = layers.Reshape(x_shape)(prev_layer)
        elif len(prev_layer.shape) == 3:
            x = prev_layer
        else:
            x_shape = (prev_layer.shape[1], 1)
            x = layers.Reshape(x_shape)(prev_layer)

        x = layers.UpSampling1D(size=size)(x)
        return x

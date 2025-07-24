from .baselayer import BaseLayer, HyperparametersManager


class UpSampling2DLayer(BaseLayer):

    def get_type(self):
        return "UpSampling2D"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("size",
                                      hp_manager.hp_upsampling_size)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        size = self.get_hyperparameter("size")

        if len(prev_layer.shape) == 4:
            x = prev_layer
        elif len(prev_layer.shape) == 3:
            x_shape = (prev_layer.shape[1], prev_layer.shape[2], 1)
            x = layers.Reshape(x_shape)(prev_layer)
        else:
            raise Exception("Not supported")

        x = layers.UpSampling2D(size=size)(x)
        return x

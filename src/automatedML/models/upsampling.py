from .baselayer import BaseLayer, HyperparametersManager


class UpSamplingLayer(BaseLayer):

    def get_type(self):
        return "UpSampling"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("size",
                                      hp_manager.hp_upsampling_size)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        size = self.get_hyperparameter("size")

        if len(prev_layer.shape) == 4:
            x = layers.UpSampling2D(size=size)(prev_layer)
        elif len(prev_layer.shape) == 3:
            x = layers.UpSampling1D(size=size)(prev_layer)
        else:
            raise Exception("Not supported")
        return x

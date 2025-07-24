from .baselayer import HyperparametersManager, BaseLayer


class FullyConnectedRNNLayer(BaseLayer):

    def get_type(self):
        return "FullyConnectedRNN"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("unit",
                                      hp_manager.hp_recurent_units)
        self._register_hyperparameter("fun",
                                      hp_manager.hp_af_names)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        unit = self.get_hyperparameter("unit")
        fun = self.get_hyperparameter("fun")

        if len(prev_layer.shape) == 3:
            x = prev_layer
        else:
            raise Exception("Not supported")

        x = layers.SimpleRNN(units=unit,
                             kernel_initializer=initializer,
                             activation=fun)(x)

        return x

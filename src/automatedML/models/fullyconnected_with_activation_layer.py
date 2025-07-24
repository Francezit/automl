from .baselayer import BaseLayer, HyperparametersManager


class FullyConnectedWithActivationLayer(BaseLayer):

    def get_type(self):
        return "FullyConnectedWithActivation"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("size",
                                      hp_manager.hp_fc_sizes)
        self._register_hyperparameter("fun",
                                      hp_manager.hp_af_names)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        size = self.get_hyperparameter("size")
        fun = self.get_hyperparameter("fun")
        return layers.Dense(
            units=size,
            kernel_initializer=initializer,
            activation=fun
        )(prev_layer)

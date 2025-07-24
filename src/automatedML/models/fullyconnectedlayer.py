from .baselayer import BaseLayer, HyperparametersManager


class FullyConnectedLayer(BaseLayer):

    def get_type(self):
        return "FullyConnected"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("size",
                                      hp_manager.hp_fc_sizes)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        size = self.get_hyperparameter("size")
        return layers.Dense(
            units=size,
            kernel_initializer=initializer,
            activation='linear'
        )(prev_layer)

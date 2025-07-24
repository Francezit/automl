from .baselayer import BaseLayer, HyperparametersManager


class ActivationLayer(BaseLayer):

    def get_type(self):
        return "Activation"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("fun",
                                      hp_manager.hp_af_names)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        fun = self.get_hyperparameter("fun")
        return layers.Activation(activation=fun)(prev_layer)





from .baselayer import BaseLayer, HyperparametersManager


class DropoutLayer(BaseLayer):

    def get_type(self):
        return "Dropout"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("prob",
                                      hp_manager.hp_dp_probs)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        prob = self.get_hyperparameter("prob")
        return layers.Dropout(rate=prob)(prev_layer)

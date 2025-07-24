from .baselayer import BaseLayer, HyperparametersManager


class AlgebraOperatorLayer(BaseLayer):
    def get_type(self):
        return "AlgebraOperator"

    def is_trainable(self) -> bool:
        return False

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("op",
                                      hp_manager.hp_op_types)
        self._register_hyperparameter("val",
                                      hp_manager.hp_op_value)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        op = self.get_hyperparameter("op")
        val = self.get_hyperparameter("val")

        if op == "none":
            return prev_layer
        elif op == "add":
            return prev_layer+val
        elif op == "multiply":
            return prev_layer*val
        else:
            raise Exception("Op not supported")
from .baselayer import BaseLayer, HyperparametersManager


class LSTMLayer(BaseLayer):

    def get_type(self):
        return "LSTM"

    def is_trainable(self) -> bool:
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("unit",
                                      hp_manager.hp_recurent_units)
        self._register_hyperparameter("fun",
                                      hp_manager.hp_af_names)

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        unit = self.get_hyperparameter("unit")
        fun = self.get_hyperparameter("fun")
        return_sequences = self.get_hyperparameter(
            "return_sequences", default=False)

        if len(prev_layer.shape) == 3:
            x = prev_layer
        else:
            raise Exception("Not supported")

        x = layers.LSTM(units=unit,
                        return_sequences=return_sequences,
                        kernel_initializer=initializer,
                        activation=fun)(x)

        return x

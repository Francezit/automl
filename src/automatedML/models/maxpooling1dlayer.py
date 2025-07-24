from .baselayer import BaseLayer


class MaxPooling1DLayer(BaseLayer):

    def get_type(self):
        return "MaxPooling1D"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        return layers.MaxPooling1D()(prev_layer)

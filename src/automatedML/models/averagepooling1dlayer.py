from .baselayer import BaseLayer


class AveragePooling1DLayer(BaseLayer):

    def get_type(self):
        return "AveragePooling1D"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        return layers.AveragePooling1D()(prev_layer)

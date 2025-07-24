from .baselayer import BaseLayer


class MaxPooling2DLayer(BaseLayer):

    def get_type(self):
        return "MaxPooling2D"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        return layers.MaxPooling2D()(prev_layer)

from .baselayer import BaseLayer


class AveragePooling2DLayer(BaseLayer):

    def get_type(self):
        return "AveragePooling2D"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        return layers.AveragePooling2D()(prev_layer)
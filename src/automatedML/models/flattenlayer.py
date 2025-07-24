from .baselayer import BaseLayer


class FlattenLayer(BaseLayer):

    def get_type(self):
        return "Flatten"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        return layers.Flatten()(prev_layer)

from .baselayer import BaseLayer


class LocalResponseNormalizationLayer(BaseLayer):

    def get_type(self):
        return "LocalResponseNormalization"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers, nn_functions

        layer = layers.Lambda(nn_functions.local_response_normalization)
        return layer(prev_layer)

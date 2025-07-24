from .baselayer import BaseLayer


class MaxPoolingLayer(BaseLayer):

    def get_type(self):
        return "MaxPooling"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        if len(prev_layer.shape) == 4:
            x = layers.MaxPooling2D()(prev_layer)

        elif len(prev_layer.shape) == 3:
            x = layers.MaxPooling1D()(prev_layer)

        else:
            raise Exception("Not supported")

        return x

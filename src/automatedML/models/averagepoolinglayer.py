from .baselayer import BaseLayer


class AveragePoolingLayer(BaseLayer):

    def get_type(self):
        return "AveragePooling"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        if len(prev_layer.shape) == 4:
            x = layers.AveragePooling2D()(prev_layer)

        elif len(prev_layer.shape) == 3:
            x = layers.AveragePooling1D()(prev_layer)

        else:
            raise Exception("Not supported")

        return x

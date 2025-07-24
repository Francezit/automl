from .baselayer import BaseLayer

class BatchNormalizationLayer(BaseLayer):

    def get_type(self):
        return "BatchNormalization"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers
        
        return layers.BatchNormalization()(prev_layer)

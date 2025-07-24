from .baselayer import BaseLayer

class IdentityLayer(BaseLayer):

    def get_type(self):
        return "Identity"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        return prev_layer

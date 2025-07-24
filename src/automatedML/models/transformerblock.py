from .baselayer import BaseLayer


# TODO to improve


class TransformerBlock(BaseLayer):
    def get_type(self):
        return "TransformerBlock"

    def is_trainable(self) -> bool:
        return True

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers
        pass

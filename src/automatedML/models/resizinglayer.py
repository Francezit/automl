from .baselayer import BaseLayer


class ResizingLayer(BaseLayer):

    def get_type(self):
        return "Resizing"

    def is_trainable(self) -> bool:
        return False

    def __init__(self, target_output=None, interpolation="bilinear", **kwargs) -> None:
        super().__init__(**kwargs)
        assert target_output,"You must specify target_output in Resizing Layer"
        self.__target_output = target_output
        self.__interpolation = interpolation

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        return layers.Resizing(
            height=self.__target_output[0],
            width=self.__target_output[1],
            interpolation=self.__interpolation
        )(prev_layer)

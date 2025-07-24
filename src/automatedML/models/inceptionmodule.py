from .baselayer import BaseLayer, HyperparametersManager


class InceptionModule(BaseLayer):

    def __init__(self, **kwargs) -> None:
        kwargs["cascade_mode"] = True

        super().__init__(**kwargs)

    def get_type(self):
        return "InceptionModule"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        pass

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        tower_1 = layers.Conv2D(
            64, (1, 1),
            padding='same', activation='relu', kernel_initializer=initializer
        )(prev_layer)
        tower_1 = layers.Conv2D(
            64, (3, 3),
            padding='same', activation='relu', kernel_initializer=initializer
        )(tower_1)
        tower_2 = layers.Conv2D(
            64, (1, 1),
            padding='same', activation='relu', kernel_initializer=initializer
        )(prev_layer)
        tower_2 = layers.Conv2D(
            64, (5, 5),
            padding='same', activation='relu', kernel_initializer=initializer
        )(tower_2)
        tower_3 = layers.MaxPooling2D(
            (3, 3),
            strides=(1, 1), padding='same', kernel_initializer=initializer
        )(prev_layer)
        tower_3 = layers.Conv2D(
            64, (1, 1),
            padding='same', activation='relu', kernel_initializer=initializer
        )(tower_3)

        return layers.concatenate(
            [tower_1, tower_2, tower_3],
            axis=3
        )

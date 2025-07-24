
from automatedML.models import HyperparametersManager, HyperparameterFormat, BaseLayer, FlattenLayer, FullyConnectedLayer, ActivationLayer, DropoutLayer, BaseSequentialBlock, FullyConnectedWithActivationLayer, BatchNormalizationLayer


class FullyConnectedCustomBlock(BaseSequentialBlock):

    def get_type(self):
        return "FullyConnectedCustomBlock"

    def is_trainable(self) -> bool:
        return True

    def get_layers_in_block(self, hp_manager: HyperparametersManager):
        return [
            FullyConnectedWithActivationLayer,
            BatchNormalizationLayer,
            DropoutLayer(prob=0.4,
                         hp_manager=hp_manager),
            ActivationLayer,
            FlattenLayer,
            ActivationLayer(fun="tanh",
                            hp_manager=hp_manager),
            FullyConnectedLayer(cascade_mode=True,
                                num_cascade_layer=2,
                                fun="tanh",
                                hp_manager=hp_manager)
        ]


class CustomFullyConnectedLayer(BaseLayer):

    def get_type(self):
        return "CustomFullyConnectedLayer"

    def is_trainable(self) -> bool:
        return True

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter(
            "size",
            hp_manager.hp_fc_sizes
        )
        self._register_hyperparameter(
            "test_1",
            HyperparameterFormat(
                name="boolean",
                values=[True, False]
            )
        )

    def _internal_compile(self, prev_layer, initializer):
        from automatedML.engine import layers

        size = self.get_hyperparameter("size")
        return layers.Dense(
            units=size,
            kernel_initializer=initializer,
            activation='linear'
        )(prev_layer)


__all__ = ["FullyConnectedCustomBlock", "CustomFullyConnectedLayer"]

import automatedML as aml


hp_manager=aml.models.get_default_hp_manager()

layers=aml.models.build_sequential_block(
    layers=[
        aml.models.FullyConnectedLayer,
        aml.models.build_parallel_block(
            layers=[
                aml.models.build_sequential_block(
                    layers=[
                        aml.models.BatchNormalizationLayer,
                        aml.models.FullyConnectedLayer,
                        aml.models.DropoutLayer
                    ],
                    hp_manager=hp_manager
                ),
                aml.models.BatchNormalizationLayer
            ],
            hp_manager=hp_manager
        ),
        aml.models.build_sequential_block(
             layers=[
                aml.models.BatchNormalizationLayer,
                aml.models.FullyConnectedLayer,
                aml.models.DropoutLayer
            ],
            hp_manager=hp_manager,
            parallel_mode=True,
            num_parallel_layer=3
        )
    ],
    hp_manager=hp_manager
)

ann = aml.ann.ANNModel(
    input_size= (12, 2),
    output_size= 1,
    type_of_task= aml.TypeOfTask.BINARY_CLASSIFICATION
)

ann.make(layers)

ann.model_plot("ciao.png")
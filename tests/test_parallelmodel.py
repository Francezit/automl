import unittest
import testcommon as tsc


class TestParallelModel(unittest.TestCase):

    def test_parallel_model_1(self):
        keys = [
            "|!FullyConnectedBlock-num_cascade_layer=3-num_parallel_layer=3",
            "£|!FullyConnectedBlock:num_cascade_layer=3:num_parallel_layer=3",
            "Activation",
            "|Convolutional1D:num_parallel_layer=3"
        ]
        parsed_model = tsc.models.parse_layers(keys, tsc.hp_manager)
        # tsc.models.set_random_hyperparameters(parsed_model)
        self.assertEqual(len(parsed_model), len(keys))

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(parsed_model)
        self.assertTrue(res)

    def test_parallel_model_2(self):
        keys = [
            tsc.models.FullyConnectedBlock(
                num_cascade_layer=3, num_parallel_layer=3, cascade_mode=True, parallel_mode=True),
            tsc.models.FullyConnectedBlock(num_cascade_layer=3, num_parallel_layer=3,
                                           cascade_mode=True, parallel_mode=True, partial_freeze_mode=True),
            tsc.models.ActivationLayer(),
            tsc.models.Convolutional1DLayer(
                num_parallel_layer=3, parallel_mode=True, merge_type="add")
        ]
        parsed_model = tsc.models.parse_layers(keys, tsc.hp_manager)
        # tsc.models.set_random_hyperparameters(parsed_model)
        self.assertEqual(len(parsed_model), len(keys))

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(parsed_model)
        self.assertTrue(res)

    def test_parallel_model_3(self):
        n = 3
        l1 = tsc.models.FullyConnectedBlock(num_cascade_layer=3,
                                            num_parallel_layer=n,
                                            partial_freeze_mode=True,
                                            cascade_mode=True,
                                            parallel_mode=True)
        l2 = tsc.models.FullyConnectedBlock(num_cascade_layer=3,
                                            cascade_mode=True,
                                            partial_freeze_mode=True)
        l1_adv = tsc.models.transform_sequential_to_parallel(l2, n_parallel=n)
        self.assertEqual(l2.count_hyperparameters*n+1,
                         l1_adv.count_hyperparameters)

    def test_parse_skipconnection_adv(self):
        keys = [
            "&FullyConnected:size=100",
            "!FullyConnected",
            "Activation",
            "&FullyConnected",
            {
                "merge_type": "concatenate",
                "skip_connection_mode": True,
                "skip_connection": "concatenate",
                "branch1": [
                    "!FullyConnected",
                    "Activation",
                ],
                "branch2": [
                    "&FullyConnected:size=100"
                ],
                "branch3": [
                    "!FullyConnectedBlock-num_cascade_layer=10"
                ]
            },
            "Flatten",
            "Activation",
        ]
        parsed_model = tsc.models.parse_layers(keys, tsc.hp_manager)
        self.assertEqual(len(parsed_model), len(keys))

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(parsed_model)
        self.assertTrue(res)
        model_ann.model_plot("./tests/plots/test_parse_skipconnection_adv.png")

    def test_parse_skipconnection(self):
        keys = [
            "&FullyConnected:size=100",
            "#FullyConnected",
            "#|FullyConnectedBlock",
            "Activation",
            "|FullyConnected",
            "Flatten",
            "Activation",
        ]
        parsed_model = tsc.models.parse_layers(keys, tsc.hp_manager)
        self.assertEqual(len(parsed_model), len(keys))

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(parsed_model)
        self.assertTrue(res)
        model_ann.model_plot("./tests/plots/test_parse_skipconnection.png")

    def test_parse_model_1(self):
        keys = [
            "&FullyConnected:size=100",
            "!FullyConnected",
            "Activation",
            "&FullyConnected",
            {
                "merge_type": "concatenate",
                "branch1": [
                    "!FullyConnected",
                    "Activation",
                ],
                "branch2": [
                    "&FullyConnected:size=100"
                ],
                "branch3": [
                    "!FullyConnectedBlock-num_cascade_layer=10"
                ]
            },
            "Flatten",
            "Activation",
        ]
        parsed_model = tsc.models.parse_layers(keys, tsc.hp_manager)
        self.assertEqual(len(parsed_model), len(keys))

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(parsed_model)
        self.assertTrue(res)
        model_ann.model_plot("./tests/plots/test_parse_model_1.png")

    def test_parse_model_2(self):
        model = [
            tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
            tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
            tsc.models.build_parallel_block([
                tsc.models.build_sequential_block([
                    tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
                    tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
                ]),
                tsc.models.build_sequential_block([
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                ])
            ]),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager)
        ]

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(model)
        self.assertTrue(res)

        model_ann.model_plot("./tests/plots/test_parse_model_2.png")

    def test_parse_model_3(self):
        model = [
            tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
            tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
            tsc.models.build_parallel_block([
                tsc.models.build_sequential_block([
                    tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
                    tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
                ]),
                tsc.models.build_sequential_block([
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                ])
            ]),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
            tsc.models.build_parallel_block([
                tsc.models.build_sequential_block([
                    tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
                    tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
                    tsc.models.DropoutLayer(hp_manager=tsc.hp_manager),
                ]),
                tsc.models.build_sequential_block([
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                    tsc.models.build_parallel_block([
                        tsc.models.FullyConnectedBlock(
                            hp_manager=tsc.hp_manager),
                        tsc.models.FullyConnectedBlock(
                            hp_manager=tsc.hp_manager),
                    ]),
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                ])
            ]),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager)
        ]

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(model)
        self.assertTrue(res)

        model_ann.model_plot("./tests/plots/test_parse_model_3.png")

    def test_parse_model_4(self):
        model = [
            tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
            tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
            tsc.models.build_with_skip_connection([
                tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
                tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
                tsc.models.IdentityLayer
            ]),
            tsc.models.build_cascade_block([
                tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                tsc.models.build_with_skip_connection([
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                    tsc.models.build_parallel_block([
                        tsc.models.FullyConnectedBlock(
                            hp_manager=tsc.hp_manager),
                        tsc.models.FullyConnectedBlock(
                            hp_manager=tsc.hp_manager),
                    ]),
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager)
                ])
            ], num_cascade_layer=4),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager)
        ]

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(model)
        self.assertTrue(res)

        model_ann.model_plot("./tests/plots/test_parse_model_4.png")

    def test_cascade_skip_parallel(self):#TODO check error
        model = [
            tsc.models.build_cascade_block([
                tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                tsc.models.build_with_skip_connection([
                    tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
                ])
            ], num_cascade_layer=4),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager)
        ]

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(model)
        self.assertTrue(res)

        model_ann.model_plot("./tests/plots/test_cascade_skip_parallel.png")


    def test_parse_cascade_mode(self):
        model = [
            tsc.models.build_cascade_block([
                tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
                tsc.models.build_with_skip_connection([
                    tsc.models.ConvolutionalLayer(hp_manager=tsc.hp_manager)
                ])
            ], num_cascade_layer=4),
            tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager)
        ]

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(model)
        self.assertTrue(res)

        model_ann.model_plot("./tests/plots/test_cascade_mode.png")

    def test_parse_skip_connection(self):
        model = [
            tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
            tsc.models.build_with_skip_connection([
                tsc.models.ConvolutionalLayer(hp_manager=tsc.hp_manager)
            ]),
            tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager)
        ]

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        res = model_ann.make(model)
        self.assertTrue(res)

        model_ann.model_plot("./tests/plots/test_skip_connection.png")

    def test_layer_key_1(self):
        keys = [
            "&FullyConnected:size=100",
            "!FullyConnected",
            "Activation",
            "&FullyConnected",
            {
                "merge_type": "concatenate",
                "branch1": [
                    "!FullyConnected",
                    "Activation",
                ],
                "branch2": [
                    "&FullyConnected:size=100"
                ],
                "branch3": [
                    "!FullyConnectedBlock-num_cascade_layer=10"
                ]
            },
            "Flatten",
            "Activation",
        ]
        parsed_model = tsc.models.parse_layers(keys, tsc.hp_manager)
        self.assertEqual(len(parsed_model), len(keys))

        layer_keys = [l.layerKey for l in parsed_model]
        layer_def = tsc.models.parse_layers(layer_keys, tsc.hp_manager)

        for d, p in zip(layer_def, parsed_model):
            self.assertEqual(d.layerKey, p.layerKey)

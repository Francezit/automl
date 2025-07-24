import unittest
import testcommon as tsc


class TestModel(unittest.TestCase):

    def test_layer_block(self):
        model = [
            tsc.models.FullyConnectedBlock(hp_manager=tsc.hp_manager),
            tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager)
        ]
        for l in model:
            l.set_random_hyperparameters()

        model_ann = tsc.ann.ANNModel(
            (12, 2), 1, tsc.TypeOfTask.BINARY_CLASSIFICATION)
        model_ann.make(model)
        model_ann.model_summary()
        self.assertTrue(model_ann.is_trainable)

    def test_clone_layer(self):
        t1 = tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager)
        t1_c = t1.clone()
        self.assertEqual(t1.layerKey, t1_c.layerKey)

        t2 = tsc.models.AutoEncoder1DBlock(hp_manager=tsc.hp_manager)
        t2_c = t2.clone()
        self.assertEqual(t2.layerKey, t2_c.layerKey)

        t3 = tsc.models.get_layer_by_key(
            "&FullyConnected:size=100", hp_manager=tsc.hp_manager)
        t3_c = t3.clone()
        self.assertEqual(t3.layerKey, t3_c.layerKey)

        t4 = tsc.models.get_layer_by_key(
            "!FullyConnectedBlock-num_cascade_layer=10", hp_manager=tsc.hp_manager)
        t4_c = t4.clone()
        self.assertEqual(t4.layerKey, t4_c.layerKey)
        pass

    def test_layer_residual_block(self):

        model = [
            tsc.models.ResidualBlock(hp_manager=tsc.hp_manager)
        ]
        model_ann = tsc.ann.ANNModel(
            (28, 28, 3), 1,  tsc.TypeOfTask.BINARY_CLASSIFICATION)
        v = model_ann.make(model)
        self.assertTrue(v)

        model = [
            tsc.models.ResidualBlock(hp_manager=tsc.hp_manager),
            tsc.models.FullyConnectedLayer(hp_manager=tsc.hp_manager),
            tsc.models.ActivationLayer(hp_manager=tsc.hp_manager)
        ]
        model_ann = tsc.ann.ANNModel(
            (28, 28, 3), 1,  tsc.TypeOfTask.BINARY_CLASSIFICATION)
        v = model_ann.make(model)
        self.assertTrue(v)

    def test_cascade(self):
        layers = [
            "Flatten",
            "!FullyConnected",
            "Activation",
            "&FullyConnected"
        ]
        model = tsc.models.parse_layers(layers, tsc.hp_manager)
        layers_2 = []
        for l in model:
            layers_2.append(l.extendedType)
        model = tsc.models.parse_layers(layers_2, tsc.hp_manager)

        model_ann = tsc.ann.ANNModel(
            (28, 28, 3), 1,  tsc.TypeOfTask.BINARY_CLASSIFICATION)
        model_ann.make(model)
        self.assertTrue(model_ann.is_trainable)

    def test_parse_layers(self):
        layers = [
            "Flatten",
            "!FullyConnected",
            "Activation",
            [
                "Flatten",
                "!FullyConnected",
                "Activation",
                "&FullyConnected"
            ],
            "&FullyConnected"
        ]
        model = tsc.models.parse_layers(layers, tsc.hp_manager)
        self.assertTrue(model is not None)

    def test_autoencodeblock(self):
        autoendored = tsc.models.get_layer_by_key(
            "AutoEncoder1DBlock", tsc.hp_manager)
        model_ann = tsc.ann.ANNModel(
            (28, 3), 1,  tsc.TypeOfTask.BINARY_CLASSIFICATION)
        model_ann.make(autoendored)
        self.assertTrue(model_ann.is_trainable)

    def test_freeze(self):
        instances1 = tsc.models.get_layer_by_key(
            "&FullyConnectedWithActivation:size=100", tsc.hp_manager)
        self.assertEqual(instances1.count_hyperparameters, 0)

        instances2 = tsc.models.get_layer_by_key(
            "£FullyConnectedWithActivation:size=100", tsc.hp_manager)
        self.assertEqual(instances2.count_hyperparameters, 1)

        code = "&FullyConnected:size=100"
        l = tsc.models.get_layer_by_key(code, tsc.hp_manager)
        self.assertEqual(code, l.extendedType)

        code1 = "&FullyConnected"
        l = tsc.models.get_layer_by_key(code1, tsc.hp_manager)
        self.assertEqual(l.layerKey, "&FullyConnected:size=512")

    def test_flags(self):
        instances3 = tsc.models.get_layers_by_class_name(
            "!£FullyConnectedWithActivation:size=100:num_cascade_layer=5", tsc.hp_manager)
        ls = tsc.models.compute_layermap(tsc.hp_manager, [
            "!£FullyConnected:num_cascade_layer=5",
            "&Activation:fun=relu",
            "Dropout",
            "BatchNormalization",
            "!£Convolutional1D:kernel=3",
            "Pooling1D"
        ])
        self.assertGreater(len(ls), 0)

    def test_flag_skipconnection(self):
        instance1 = tsc.models.get_layers_by_class_name(
            "#&FullyConnected:size=100", tsc.hp_manager)
        self.assertGreaterEqual(len(instance1), 1)
        instance2 = tsc.models.get_layers_by_class_name(
            "#FullyConnected", tsc.hp_manager)
        self.assertGreaterEqual(len(instance2), 1)
        instance3 = tsc.models.get_layers_by_class_name(
            "#|FullyConnected", tsc.hp_manager)
        self.assertGreaterEqual(len(instance3), 1)

        model = tsc.models.parse_layers([
            "#|FullyConnected-skip_connection=concatenate-num_parallel_layer=10",
        ], tsc.hp_manager)

        model_ann = tsc.ann.ANNModel(
            (28, 28, 3), 1,  tsc.TypeOfTask.BINARY_CLASSIFICATION)
        model_ann.make(model)
        self.assertTrue(model_ann.is_trainable)
        model_ann.model_plot("tests/plots/test_flag_skipconnection.png")
        pass

    def test_layer_key(self):
        instances1 = tsc.models.get_layers_by_class_name(
            "&FullyConnected:size=100", tsc.hp_manager)
        instances2 = tsc.models.get_layers_by_class_name(
            "&FullyConnected:size=100", tsc.hp_manager, True)
        instances3 = tsc.models.get_layers_by_class_name(
            "&FullyConnectedWithActivation:size=100", tsc.hp_manager)
        instances4 = tsc.models.get_layers_by_class_name(
            "&FullyConnectedWithActivation:size=100", tsc.hp_manager, True)
        instances5 = tsc.models.get_layers_by_class_name(
            "£FullyConnectedWithActivation:fun=elu", tsc.hp_manager, True)
        instances6 = tsc.models.get_layers_by_class_name(
            "£FullyConnectedWithActivation:fun=elu", tsc.hp_manager)
        instances = tsc.models.compute_layermap(tsc.hp_manager, [
            "Activation",
            "Flatten",
            "!ResidualBlock",
            "&FullyConnected:size=100",
            "£FullyConnectedWithActivation:fun=elu-size=50"
        ])

        metadata1 = tsc.models.get_layer_by_key(
            "&FullyConnected:size=100", tsc.hp_manager).metadata
        layer1 = tsc.models.get_layer_by_metadata(metadata1, tsc.hp_manager)

        metadata2 = tsc.models.get_layer_by_key(
            "£FullyConnectedWithActivation:fun=elu", tsc.hp_manager).metadata
        layer2 = tsc.models.get_layer_by_metadata(metadata2, tsc.hp_manager)
        layer2.freeze()
        self.assertFalse(layer2.has_hyperparameters)
        pass

    def test_import_layer(self):
        filename = "tests/custom_layer.py"
        n = tsc.models.load_custom_layers(filename)
        self.assertEqual(len(n), 2)

        maps = tsc.models.compute_layermap(tsc.hp_manager, [
            "CustomFullyConnectedLayer",
            "!FullyConnectedCustomBlock"
        ])
        self.assertGreater(len(maps), 0)

        model = tsc.models.parse_layers([
            "!£CustomFullyConnectedLayer:num_cascade_layer=5",
        ], tsc.hp_manager)

        model_ann = tsc.ann.ANNModel(
            (28, 28, 3), 1,  tsc.TypeOfTask.BINARY_CLASSIFICATION)
        model_ann.make(model)
        self.assertTrue(model_ann.is_trainable)

        pass

    def test_stat_layermap(self):

        layer_map = tsc.models.compute_layermap_new(
            tsc.hp_manager,
            [
                "Activation",
                "Flatten",
                "!ResidualBlock",
                "&FullyConnected:size=100",
                "£FullyConnectedWithActivation:fun=elu-size=50"
            ]
        )

        layer_map = tsc.models.compute_layermap_new(
            tsc.hp_manager,
            [
                "AutoEncoder1DBlock",
                "AutoEncoder2DBlock"
            ]
        )

        pass

        '''
        class_names = tsc.models.get_all_class_names()

        classes_test = []
        for i, c in enumerate(class_names):
            classes_test.append(c)

            layer_map = tsc.models.compute_layermap(
                tsc.hp_manager,
                classes_test)

            self.assertTrue(layer_map is not None)
            dt = tsc.models.get_statistic_from_layermap(
                layer_map,
                tsc.hp_manager
            )

            print(c)
            print(dt.to_string())
        '''

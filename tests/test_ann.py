import unittest
import testcommon as tsc


class TestAnn(unittest.TestCase):
    def test_ann_1(self):
        net = tsc.ann.define_architecture_by_layers(
            layers=[
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
            ],
            name="custom_ann"
        )
        self.assertTrue(net is not None)
        hp_vect = net.get_hp_vector()
        self.assertTrue(len(hp_vect) > 0)
        hp_formats = net.get_hp_formats()
        self.assertTrue(len(hp_formats) > 0)

    def test_ann_2(self):

        net = tsc.ann.define_architecture_by_layers(
            layers=[
                tsc.models.FlattenLayer,
                "!FullyConnected",
                tsc.models.ActivationLayer(hp_manager=tsc.hp_manager),
                [
                    "Flatten",
                    "!FullyConnected",
                    tsc.models.ActivationLayer(
                        fun="linear", freeze_mode=True),
                    "&FullyConnected"
                ],
                {
                    "merge_type": "concatenate",
                    "branch1": [
                        "!FullyConnected",
                        tsc.models.ActivationLayer,
                    ],
                    "branch2": [
                        tsc.models.FullyConnectedWithActivationLayer(
                            size=100,
                            partial_freeze_mode=True
                        )
                    ],
                    "branch3": [
                        "!FullyConnectedBlock-num_cascade_layer=10"
                    ],
                    "branch4": [
                        "Identity"
                    ]
                },
                "Flatten",
                "&FullyConnected"
            ],
            name="custom_ann"
        )
        self.assertTrue(net is not None)
        hp_vect = net.get_hp_vector()
        self.assertTrue(len(hp_vect) > 0)
        hp_formats = net.get_hp_formats()
        self.assertTrue(len(hp_formats) > 0)

    def test_ann_3(self):
        net = tsc.ann.define_architecture_by_layers(
            layers=[
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
            ],
            name="custom_ann"
        )

        f = "tests/anns/test_ann_3.json"
        tsc.ann.save_architecture(f, net)
        net1 = tsc.ann.load_architecture(f)

        for x, y in zip(net.to_model(), net1.to_model()):
            self.assertEqual(x.layerKey, y.layerKey)

    def test_ann_4(self):
        net = tsc.ann.define_architecture_by_layers([
            tsc.models.build_sequential_block([
                tsc.models.FullyConnectedLayer,
                tsc.models.BatchNormalizationLayer,
                tsc.models.ActivationLayer
            ], cascade_mode=True, num_cascade_layer=3)
        ])

        f = "tests/anns/test_ann_4.json"
        tsc.ann.save_architecture(f,net)
        net2=tsc.ann.load_architecture(f)

        netmodel1:tsc.ann.ANNModel=tsc.ann.create_ann_by_architecture(net,**tsc.ann_params)
        netmodel1.model_plot("tests/anns/test_ann_4_target.png")
        netmodel2:tsc.ann.ANNModel=tsc.ann.create_ann_by_architecture(net2,**tsc.ann_params)
        netmodel2.model_plot("tests/anns/test_ann_4_original.png")

        self.assertListEqual(net.to_sequence(),net2.to_sequence())

        pass

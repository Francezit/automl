import unittest
import testcommon as tsc


class TestAnnModel(unittest.TestCase):

    def test_load_and_hp(self):
        model = tsc.annmodels.get_architecture("fcnn")
        hp = model.get_hp_vector()
        hp_format = model.get_hp_formats()
        hp[-1] = hp_format[-1].get_random_value()
        model.set_hp_vector(hp)
        self.assertTrue(model is not None)

    def test_load(self):
        net1 = tsc.annmodels.get_architecture("fcnn")
        self.assertTrue(net1 is not None)
        net3 = tsc.annmodels.get_architecture("cnn")
        self.assertTrue(net3 is not None)

        models = [net1.to_model(),  net3.to_model()]
        for model in models:
            annmodel = tsc.ann.ANNModel(**tsc.ann_params)
            v = annmodel.make(model)
            if not v:
                print(annmodel.freeze_error)
            self.assertTrue(v)
            l1=annmodel.train(**tsc.train_params)
            l2=annmodel.eval(**tsc.eval_params)
            self.assertTrue(annmodel.is_ready)
            pass
        pass

    def test_all_models(self):

        models=[
            tsc.annmodels.get_architecture(name)
            for name in tsc.annmodels.get_supported_models()
        ]
        self.assertTrue(all([x is not None for x in models]))

    def test_convnextblock(self):
        nn=tsc.annmodels.get_architecture("convnext")

        pass

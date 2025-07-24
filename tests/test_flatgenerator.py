import unittest
import testcommon as tsc


class TestFlatModelGenerator(unittest.TestCase):
    def test_generate_randomly(self):
        generator = tsc.FlatGenerator(data=tsc.test_dataset,
                                      settings=tsc.generator_setting)
        i = 0
        while i < 10:
            model, metric = generator.generate_randomly()
            if model.is_empty:
                i = i+1
            else:
                model.model_summary()
                full_metric = generator.fit(model)
                self.assertTrue(True)
                return
        self.assertTrue(False)

    def test_generate_iteratively(self):
        i = 0
        generator = tsc.FlatGenerator(data=tsc.test_dataset,
                                      settings=tsc.generator_setting)
        while i < 10:

            model, metric = generator.generate_iteratively()
            if model.is_empty:
                i = i+1
            else:
                model.model_summary()
                full_metric = generator.fit(model)
                self.assertTrue(model is not None)
                return
        self.assertTrue(False)

    pass

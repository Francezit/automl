import unittest
import testcommon as tsc


class TestModelEvaluation(unittest.TestCase):

    def test_eval_model(self):
        results = tsc.evaluation_models(configurations=[tsc.template_ann_1],
                                        ann_args=tsc.ann_params,
                                        eval_args=tsc.eval_params,
                                        train_args=tsc.train_params,
                                        num_workers=1)
        print(results)
        self.assertTrue(results is not None)

    def test_eval_models(self):
        results = tsc.evaluation_models(configurations=[tsc.template_ann_1, tsc.template_ann_2, tsc.template_ann_3],
                                        ann_args=tsc.ann_params,
                                        eval_args=tsc.eval_params,
                                        train_args=tsc.train_params,
                                        num_workers=2)
        print(results)
        self.assertTrue(results is not None)


if __name__ == '__main__':
    unittest.main()

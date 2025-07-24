import unittest
import testcommon as tsc


class TestModelOptimizator(unittest.TestCase):
    def test_optimizator(self):
        optimizator = tsc.HyperparametersOptimizator(
            data=tsc.test_dataset,
            settings=tsc.optimizator_setting,
            seed=1234
        )

        results = optimizator.optimize(
            alg_name="ls",
            neiborh_size=5,
            num_pertubations=2
        )

        self.assertTrue(results is not None)

    def test_optimizator_rs(self):
        optimizator = tsc.HyperparametersOptimizator(
            data=tsc.test_dataset,
            settings=tsc.optimizator_setting,
            seed=1234
        )

        results = optimizator.optimize(
            alg_name="random"
        )

        self.assertTrue(results is not None)

    def test_optimizator_aco(self):
        optimizator = tsc.HyperparametersOptimizator(
            data=tsc.test_dataset,
            settings=tsc.optimizator_setting,
            seed=1234
        )

        results = optimizator.optimize(
            alg_name="aco",
            alph=1,
            beta=1,
            max_path_length=4,
            max_neighborhood_size=2,
            evaporation_rate=0.01,
            n_ants=1,
            Q=10
        )

        self.assertTrue(results is not None)

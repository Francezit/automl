import unittest
import testcommon as tsc


class TestHyperparamsOptimization(unittest.TestCase):

    def _get_common_params(self):
        return {
            "current_layers": tsc.template_ann_1,
            "ann_args": tsc.ann_params,
            "eval_args": tsc.eval_params,
            "train_args": tsc.train_params,
            "train_args_bounds": tsc.train_args_bounds,
            "timeout": 60,
            "compare_model_function": tsc.opt_algs.default_compare_model_function,
            "num_workers": None
        }

    def test_local_search(self):
        output = tsc.opt_algs.optimize_with_localsearch(
            neiborh_size=5,
            num_pertubations=2,
            **self._get_common_params()
        )

        self.assertTrue(output is not None)
        pass

    def test_bayesian(self):
        iterations = 5
        output = tsc.opt_algs.optimize_with_bayesianopt(
            **self._get_common_params()
        )

        self.assertTrue(output is not None)
        pass

    def test_aco(self):
        output = tsc.opt_algs.optimize_with_antcolonyoptimization(
            alpha=1,
            beta=1,
            max_path_length=10,
            evaporation_rate=0.01,
            max_neighborhood_size=2,
            n_ants=2,
            Q=100,
            **self._get_common_params()
        )

        self.assertTrue(output is not None)
        pass

    def test_pso(self):
        output = tsc.opt_algs.optimize_with_particle_swarm_optimization(
            n_particles=10,
            cognitive=0.5,
            social=0.3,
            inertia=0.9,
            **self._get_common_params())

        self.assertTrue(output is not None)
        pass

    def test_gridsearch(self):
        return
        num_workers = None
        output = tsc.opt_algs.optimize_with_gridsearch(current_layers=tsc.template_ann_1,
                                                       ann_args=tsc.ann_params,
                                                       eval_args=tsc.eval_params,
                                                       train_args=tsc.train_params,
                                                       train_args_bounds=tsc.train_args_bounds,
                                                       compare_model_function=tsc.optimizator.default_compare_model_function,
                                                       num_workers=num_workers)

        self.assertTrue(output is not None)
        pass

    def test_randomsearch(self):
        num_workers = None
        output = tsc.opt_algs.optimize_with_randomsearch(
            **self._get_common_params())

        self.assertTrue(output is not None)
        pass

    def test_iterated_local_search(self):
        ls_neiborh_size = 5
        num_pertubations = 2
        output = tsc.opt_algs.optimize_with_iteretedlocalsearch(
            neiborh_size=ls_neiborh_size,
            num_pertubations=num_pertubations,
            **self._get_common_params()
        )

        self.assertTrue(output is not None)
        pass

    def test_tabu_search(self):
        ls_neiborh_size = 5
        tabu_list_size = 3
        output = tsc.opt_algs.optimize_with_tabuseach(
            neiborh_size=ls_neiborh_size,
            tabu_list_size=tabu_list_size,
            num_pertubations=2,
            **self._get_common_params()
        )

        output.export("data.csv")
        #output.plot("test.svg")
        self.assertTrue(output is not None)
        pass

    def test_genetic_algorithm(self):
        population = 5
        crossover = 0.9
        mutation = 0.5
        output = tsc.opt_algs.optimize_with_geneticalgorithm(
            population_size=population,
            crossover_prob=crossover,
            num_pertubations=2,
            mutation_prob=mutation,
            **self._get_common_params()
        )

        self.assertTrue(output is not None)
        pass

    def test_hybrid_genetic_algorithm(self):
        population = 5
        crossover = 0.9
        mutation = 0.5
        output = tsc.opt_algs.optimize_with_hybridgeneticalgorithm(
            population_size=population,
            crossover_prob=crossover,
            mutation_prob=mutation,
            neiborh_size=2,
            num_pertubations=2,
            **self._get_common_params()
        )

        self.assertTrue(output is not None)
        pass

    def test_simulated_annealing(self):

        output = tsc.opt_algs.optimize_with_simulatedannelaling(
            step_size=2,
            temp=10,
            **self._get_common_params()
        )
        self.assertTrue(output is not None)

    def test_gaml(self):
        population = 5
        output = tsc.opt_algs.optimize_with_geneticalgorithm_with_machinelearning(
            population_size=population,
            num_pertubations=2,
            crossover_params=[0.6, 0.2, 10],
            mutation_params=[0.4, 0.2, 10],
            buffer_size=4,
            **self._get_common_params()
        )

        self.assertTrue(output is not None)
        pass

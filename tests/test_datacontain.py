import unittest
import testcommon as tsc


class TestDataContainer(unittest.TestCase):

    def test_minmax_normalization(self):
        test_dataset = tsc.load_data_container({
            "name": "android-malware-detection",
            "reduction_factor": 1,
            "source": "random",
            "type_of_task": "regression",
            "use_balancing": True,
            "standardization":"min-max"
        })
        self.assertTrue(True)

    def test_scale_normalization(self):
        test_dataset = tsc.load_data_container({
            "name": "android-malware-detection",
            "reduction_factor": 1,
            "source": "random",
            "type_of_task": "regression",
            "use_balancing": True,
            "standardization":"scale"
        })
        self.assertTrue(True)

    

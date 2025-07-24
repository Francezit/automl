__author__ = "Francesco Zito"
__copyright__ = "Copyright 2024, Francesco Zito"
__version__ = "0.6.0"
__email__ = "zitofrancesco@outlook.com"


from .utils import MetricInfo
from .datacontainer import *
from .component import TrainSettings,compute_train_and_eval_params

from . import ann
from . import optimizator
from . import models
from . import flatgenerator
from . import annmodels
from .internal import TypeOfTask, ProblemType


def create_logger(log_folder: str, name: str = "autoML", verbose: bool = False):
    import logging
    import os
    import time
    import sys

    os.makedirs(log_folder, exist_ok=True)
    logFilename = os.path.join(
        log_folder,
        f"{time.strftime('%Y%m%d-%H%M%S')}.txt"
    )
    logLevel = logging.DEBUG if verbose else logging.INFO

    logger = logging.Logger(name, logLevel)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logFilename)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger


def __init_module():
    # internal configuration
    import os
    import logging
    import multiprocessing

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["KMP_AFFINITY"] = "noverbose"

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('h5py').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

    # import tensorflow as tf
    # tf.debugging.disable_traceback_filtering()
    # tf.debugging.experimental.disable_dump_debug_info()
    # tf.keras.utils.disable_interactive_logging()

    # tf.get_logger().setLevel('ERROR')
    # tf.autograph.set_verbosity(3)

    multiprocessing.set_start_method('spawn', force=True)

    # def set_seed(seed_number: int):
    #   import numpy
    #   import random
    #    random.seed(seed_number)
    #    numpy.random.seed(seed_number)
    # tf.random.set_seed(seed_number)


__init_module()

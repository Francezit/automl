from .tempfolder import generate_temp_folder, create_temp_context, remove_temp_folder, TempContext
from .methods import train_model, train_model_with_kfold
from keras import Model, initializers, optimizers as training_optimizers, callbacks as training_callbacks
from keras.utils import plot_model, to_categorical
from keras.models import load_model
from tensorflow import reduce_prod, pad, constant, random, shape, floor, Variable, ones,expand_dims,reshape
from tensorflow import nn as nn_functions
from tensorflow import __version__ as tf_version
from keras import __version__ as keras_version

random.set_seed(1234)


def __check_module(name: str):
    import importlib
    exists = importlib.util.find_spec(name) is not None
    return exists


if keras_version.startswith("2.13"):
    from keras.utils import disable_interactive_logging
    from keras.backend import clear_session
    

    disable_interactive_logging()
    clear_session()
    

from keras.config import enable_unsafe_deserialization
enable_unsafe_deserialization()
del enable_unsafe_deserialization

if __check_module("tensorflow_models"):
    try:
        from tensorflow_models.vision import layers as advanced_layers
    except:
        advanced_layers = None
        print("advanced_layers models not found")


dataset_sources = [
    "keras", "random"
]
if __check_module("tensorflow_datasets"):
    dataset_sources.append("tensorflow_datasets")


def save_model(model: Model, filename: str):
    model.save(filename, overwrite=True)


def load_dataset(source: str, dataset_name: str):
    assert source in dataset_sources, "Source not supported"
    if source == "keras":
        import keras.datasets as keras_datasets
        (X_train, y_train), (X_test, y_test) = eval(
            f"keras_datasets.{dataset_name}.load_data()"
        )
        return X_train, y_train, X_test, y_test
    elif source == "tensorflow_datasets":
        import tensorflow_datasets as tfds
        import numpy as np

        ds = tfds.load(name=dataset_name,
                       as_supervised=True)
        train, test = ds["train"], ds["test"]

        train = list(zip(*train))
        X_train, y_train = np.array(train[0]), np.array(train[1])
        del train

        test = list(zip(*test))
        X_test, y_test = np.array(test[0]), np.array(test[1])
        del test

        return X_train, y_train, X_test, y_test
    elif source == "random":
        from sklearn import datasets
        import numpy as np

        n_samples = 200
        noise = 0.3
        seed = 1234
        test_split = 0.3

        X, y = datasets.make_moons(n_samples, noise=noise, random_state=seed)

        index = int(np.floor(n_samples*(1-test_split)))
        X_train = X[:index, :]
        y_train = y[:index]
        X_test = X[index:, :]
        y_test = y[index:]

        del X, y
        return X_train, y_train, X_test, y_test
    else:
        raise Exception("DATA Not supported")


def get_layer_function(name: str):
    return eval(f"layers.{name}")


def create_zip(filename: str, files: list[str]):
    import zipfile
    import os

    with zipfile.ZipFile(filename, "w") as archive:
        for f in files:
            archive.write(f, os.path.basename(f))


def extract_zip(filename: str, output_folder: str):
    import zipfile

    with zipfile.ZipFile(filename, "r") as archive:
        archive.extractall(output_folder)

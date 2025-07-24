from automatedML import DataContainer, load_data_container, TypeOfTask, ProblemType
from automatedML.evalmodel import evaluation_models
import automatedML.models as models
import automatedML.annmodels as annmodels
from automatedML.optimizator import algorithms as opt_algs, HyperparametersOptimizator, OptimizatorSettings
import automatedML.ann as ann
from automatedML.flatgenerator import FlatGenerator, FlatGeneratorSettings

test_dataset = load_data_container({
    "name": "android-malware-detection",
            "reduction_factor": 1,
            "source": {
                "trainX": "./tests/datasets/trainX-custom.csv",
                "trainY": "./tests/datasets/trainY-custom.csv",
                "testX": "./tests/datasets/testX-custom.csv",
                "testY": "./tests/datasets/testY-custom.csv"
            },
    "type_of_task": TypeOfTask.BINARY_CLASSIFICATION,
    "use_balancing": True
})


ann_params = {
    "input_size": test_dataset.input_size,
    "output_size": test_dataset.output_size,
    "type_of_task": test_dataset.type_of_task,
    "seed": 1234
}

X_train, y_train = test_dataset.get_trainingset()
train_params = {
    "X_train": X_train,
    "y_train": y_train,
    "batch_size": 32,
    "epochs": 5,
    "optimizer": "adam",
    "optimizer_configs": {
        "learning_rate": 0.001
    },
    "shuffle": True,
    "num_workers": 4,
    "loss_function_name": None
}

X_test, y_test = test_dataset.get_testset()
eval_params = {
    "X_test": test_dataset._X_test,
    "y_test": test_dataset._y_test
}

hp_manager = models.get_default_hp_manager()
hp_manager.set_hp_format(
    hp_format=models.HyperparameterFormat(
        name="hp_af_names",
        values=["relu", "tanh"]
    )
)

template_ann_1 = [
    models.FullyConnectedLayer(hp_manager=hp_manager),
    models.ActivationLayer(hp_manager=hp_manager),
    models.DropoutLayer(hp_manager=hp_manager)
]

template_ann_2 = [
    models.FullyConnectedLayer(hp_manager=hp_manager),
    models.ActivationLayer(hp_manager=hp_manager),
    models.FullyConnectedLayer(hp_manager=hp_manager),
    models.ActivationLayer(hp_manager=hp_manager),
]

template_ann_3 = [
    models.FullyConnectedLayer(hp_manager=hp_manager),
    models.ActivationLayer(hp_manager=hp_manager),
    models.FullyConnectedLayer(hp_manager=hp_manager),
    models.ActivationLayer(hp_manager=hp_manager),
    models.DropoutLayer(hp_manager=hp_manager)
]

train_args_bounds = {
    "optimizer": ["adam", "sgd", "rmsprop"],
    "optimizer_configs":
    {
        "learning_rate": [0.001, 0.01, 0.1]
    },
    "epochs": [5, 10, 20, 40, 50, 100]
}

# load generator
generator_setting = FlatGeneratorSettings()
generator_setting.layer_classes = [
    "!FullyConnected",
    "&Activation:fun=relu",
    "£Convolutional1D:kernel=3"
]

generator_setting.alg_hp_optimization_prob = 1
generator_setting.alg_max_episodes = 3

generator_setting.dyann_max_deph_size = 3

generator_setting.train_settings.epochs = 2
generator_setting.train_settings.fast_train_data_reduction_factor = 0.5

generator_setting.hp_optimization_number_workers = None
generator_setting.hp_optimization_alg = "ls"
generator_setting.hp_optimization_timeout = 120

generator_setting.hp_settings = {
    "hp_af_names": ["relu", "tanh"]
}

# load optimizator
optimizator_setting = OptimizatorSettings()
optimizator_setting.model = [
    "FullyConnected",
    "Activation",
    "Dropout",
    "FullyConnected",
    "Activation",
    "FullyConnected"
]
optimizator_setting.train_settings.num_workers = 8
optimizator_setting.train_settings_bounds = {
    "batch_size": [32, 64],
    "shuffle": [True, False],
    "optimizer": ["adam", "sgd", "rmsprop"],
    "optimizer_configs":
    {
        "learning_rate": [0.001, 0.01, 0.1]
    },
    "epochs": [2, 5, 10, 20, 40, 50, 100]
}
optimizator_setting.hp_settings = {
    "hp_af_names": ["relu", "tanh"]
}

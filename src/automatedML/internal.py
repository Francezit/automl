from enum import Enum

'''
TYPE_OF_TASK_REGRESSION = "regression"
TYPE_OF_TASK_MULTI_CLASSIFICATION = "classification"
TYPE_OF_TASK_BINARY_CLASSIFICATION = "binary-classification"
TYPE_OF_TASK_TS_FORECASTING = "ts-forecasting"
TYPE_OF_TASKS = [
    TYPE_OF_TASK_REGRESSION,
    TYPE_OF_TASK_MULTI_CLASSIFICATION,
    TYPE_OF_TASK_BINARY_CLASSIFICATION,
    TYPE_OF_TASK_TS_FORECASTING
]
'''


class TypeOfTask(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary-classification"
    TS_FORECASTING = "ts-forecasting"
    TS_MULTI_FORECASTING = "ts-multi-forecasting"
    TS_CUSTOM_FORECASTING = "ts-custom-forecasting"

    @property
    def is_continue_task(self):
        return self in [TypeOfTask.REGRESSION, TypeOfTask.TS_FORECASTING, TypeOfTask.TS_MULTI_FORECASTING,TypeOfTask.TS_CUSTOM_FORECASTING]

    @property
    def is_classification(self):
        return self in [TypeOfTask.CLASSIFICATION, TypeOfTask.BINARY_CLASSIFICATION]

    @property
    def is_forecasting(self):
        return self in [TypeOfTask.TS_FORECASTING, TypeOfTask.TS_MULTI_FORECASTING, TypeOfTask.TS_CUSTOM_FORECASTING]

    @property
    def is_discrete_task(self):
        return self == TypeOfTask.CLASSIFICATION

    @property
    def is_binary_task(self):
        return self == TypeOfTask.BINARY_CLASSIFICATION


'''
PROBLEM_TYPE_MAX = "max"
PROBLEM_TYPE_MIN = "min"
PROBLEM_TYPES = [
    PROBLEM_TYPE_MAX,
    PROBLEM_TYPE_MIN
]
'''


class ProblemType(Enum):
    MAX = "max"
    MIN = "min"


__type_of_task_info = {
    TypeOfTask.TS_FORECASTING: {
        "problem_type": ProblemType.MIN,
        "metrics": ["mae"],
        "loss": "mae"
    },
    TypeOfTask.TS_MULTI_FORECASTING: {
        "problem_type": ProblemType.MIN,
        "metrics": ["mae"],
        "loss": "mae"
    },
     TypeOfTask.TS_CUSTOM_FORECASTING: {
        "problem_type": ProblemType.MIN,
        "metrics": ["mae"],
        "loss": "mae"
    },
    TypeOfTask.BINARY_CLASSIFICATION: {
        "problem_type": ProblemType.MAX,
        "metrics": ["accuracy"],
        "loss": "binary_crossentropy"
    },
    TypeOfTask.CLASSIFICATION: {
        "problem_type": ProblemType.MAX,
        "metrics": ["accuracy"],
        "loss": "categorical_crossentropy"
    },
    TypeOfTask.REGRESSION: {
        "problem_type": ProblemType.MIN,
        "metrics": ["mae"],
        "loss": "mae"
    }
}


def get_problem_type(type_of_task: TypeOfTask) -> ProblemType:
    return __type_of_task_info[type_of_task]["problem_type"]


def get_training_metrics(type_of_task: TypeOfTask):
    item = __type_of_task_info[type_of_task]
    loss: str = item["loss"]
    metrics: str = item["metrics"]

    return loss, metrics


def get_number_metrics(type_of_task: TypeOfTask):
    item = __type_of_task_info[type_of_task]
    return len(item["metrics"])


def advanced_dic_update(*dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

from enum import Enum
import random
import numpy as np


class HyperparameterValueType(Enum):
    Float = "float"
    Boolean = "boolean"
    Integer = "int"
    Categorical = "categorical"


_hp_type_map = {
    HyperparameterValueType.Float: float,
    HyperparameterValueType.Boolean: bool,
    HyperparameterValueType.Integer: int
}


class HyperparameterFormat():
    __type: HyperparameterValueType
    __values: list
    __range: tuple[float | int]
    __default: object
    __name: str

    @property
    def default_value(self):
        return self.__default

    @property
    def name(self):
        return self.__name

    @property
    def value_type(self):
        if self.__type == HyperparameterValueType.Categorical:
            return type(self.__values[0])
        else:
            return _hp_type_map[self.__type]

    @property
    def size(self):
        return len(self.get_discrete_values())

    @property
    def type(self):
        return self.__type

    @property
    def is_discrete(self):
        return self.__type == HyperparameterValueType.Categorical

    @property
    def is_continue(self):
        return self.__type != HyperparameterValueType.Categorical

    def __init__(self, name: str, values: list = None, range: tuple[float | int] = None,
                 type: HyperparameterValueType | str = HyperparameterValueType.Categorical, default_value=None) -> None:

        if not isinstance(type, HyperparameterValueType):
            type = HyperparameterValueType(type)

        if type == HyperparameterValueType.Categorical:
            if values is None or len(values) == 0:
                raise Exception("You must indicate the values in Categorical")
        elif type != HyperparameterValueType.Boolean:
            if range is None or len(range) < 1 or range[0] >= range[1]:
                raise Exception(
                    "You must indicate the range in not Categorical type")

        assert name is not None
        self.__name = name
        self.__type = type
        self.__values = values
        self.__range = range

        if default_value:
            if not self.check_constraints(default_value):
                raise Exception("Default value out of the range")
            self.__default = default_value
        elif type == HyperparameterValueType.Categorical:
            self.__default = values[0]
        elif type == HyperparameterValueType.Boolean:
            self.__default = False
        else:
            self.__default = self.check_type(
                (self.__range[0]+self.__range[1])/2
            )

    def get_random_value(self):
        if self.__type == HyperparameterValueType.Categorical:
            return random.choice(self.__values)
        elif self.__type == HyperparameterValueType.Boolean:
            return random.choice([True, False])
        elif self.__type == HyperparameterValueType.Float:
            return random.uniform(self.__range[0], self.__range[1])
        elif self.__type == HyperparameterValueType.Integer:
            return random.randint(self.__range[0], self.__range[1])
        else:
            raise Exception("HyperparameterValueType not supported")

    def check_type(self, value):
        target_type = self.value_type
        if not isinstance(value, target_type):
            value = target_type(value)
        return value

    def check_constraints(self, value):
        if self.__type == HyperparameterValueType.Categorical:
            return value in self.__values
        elif self.__type == HyperparameterValueType.Boolean:
            return value == True or value == False
        elif value >= self.__range[0] and value <= self.__range[1]:
            return True
        else:
            return False

    def get_discrete_values(self, max_values: int = None):
        if self.__type == HyperparameterValueType.Categorical:
            n = len(self.__values)
            if max_values is None or max_values < n:
                return self.__values.copy()
            else:
                step = np.ceil(n/max_values)
                return [self.__values[i] for i in range(0, n, step)]
        elif self.__type == HyperparameterValueType.Boolean:
            return [False, True]
        else:
            nptype = np.float32 if self.__type == HyperparameterValueType.Float else np.int32
            pytype = float if self.__type == HyperparameterValueType.Float else int
            if max_values is None:
                if len(self.__range) >= 3:
                    step = nptype(self.__range[2])
                else:
                    step = (self.__range[1]-self.__range[0])/10
            else:
                step = (self.__range[1]-self.__range[0])/max_values

            if self.__type == HyperparameterValueType.Integer:
                step = nptype(np.ceil(step))

            v = np.arange(self.__range[0], self.__range[1], step, nptype)
            return [pytype(x) for x in set(v)]

    def clone(self):
        return HyperparameterFormat(self.__values, self.__range, self.__type)

    def to_dict(self):
        return {
            "name": self.__name,
            "values": self.__values,
            "range": self.__range,
            "type": self.__type.value,
            "default_value": self.__default
        }


class HyperparametersManager():
    __hp_formats: dict[str, HyperparameterFormat]

    @property
    def counts(self):
        return len(self.__hp_formats)

    def __init__(self, config: list = None):
        self.__hp_formats = dict()
        if config is not None:
            for item in config:
                self.set_hp_format(item)

    def get_hp_codes(self):
        return list(self.__hp_formats.keys())

    def set_hp_format(self, hp_format: list | tuple | HyperparameterFormat | dict):
        assert hp_format

        if isinstance(hp_format, list) or isinstance(hp_format, tuple):
            hp_format = HyperparameterFormat(*hp_format)
        elif isinstance(hp_format, dict):
            hp_format = HyperparameterFormat(**hp_format)
        elif not isinstance(hp_format, HyperparameterFormat):
            raise Exception(
                "Values not supported, it must be HyperparameterFormat")

        self.__hp_formats[hp_format.name.lower()] = hp_format

    def get_hp_format(self, hp_code: str) -> HyperparameterFormat:
        v = self.__hp_formats[hp_code.lower()]
        return v

    def clone(self):
        return HyperparametersManager(self.__hp_formats.values())

    def update(self, hp_formats: list[HyperparameterFormat]):
        hp_manager = HyperparametersManager(self.__hp_formats.values())
        for hpf in hp_formats:
            hp_manager.set_hp_format(hpf)
        return hp_manager

    def __getattr__(self, hp_code: str):
        return self.__hp_formats[hp_code.lower()]


def get_default_hp_manager():
    hp_manager = HyperparametersManager()

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_active",
            type=HyperparameterValueType.Boolean
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_fc_sizes",
            range=(25, 1000, 50),
            type=HyperparameterValueType.Integer
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_af_names",
            values=[
                "relu", "linear", "elu",
                "tanh", "gelu", "selu"
            ]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_conv_filters",
            values=[8, 16, 32, 64, 128, 256]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_conv_pooling_type",
            values=["max", "avg"]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_conv_kernel",
            values=[2, 3, 5, 7, 11]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_dp_probs",
            range=(0, 0.9),
            type=HyperparameterValueType.Float
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_recurent_units",
            values=[16, 32, 64, 128, 256, 512, 1024, 2048],
            type=HyperparameterValueType.Categorical
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_recurrent_types",
            values=["LSTM", "GRU", "FC","BiLSTM"],
            type=HyperparameterValueType.Categorical
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_residual_filters",
            values=[32, 64, 128, 256]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_residual_kernel",
            values=[2, 3]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_residual_af_names",
            values=["relu", "tanh"]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_num_cascade_layers",
            range=(1, 11),
            type=HyperparameterValueType.Integer,
            default_value=1
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_num_parallel_layer",
            range=(1, 5),
            type=HyperparameterValueType.Integer,
            default_value=1
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_upsampling_size",
            values=[2, 4, 6, 8, 10]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_merge_types",
            values=[
                "concatenate", "average", "maximum",
                "minimum", "add", "subtract", "multiply"
            ]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_op_types",
            values=[
                "none", "add", "multiply"
            ]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_op_value",
            range=(-2.0, 2.0),
            type=HyperparameterValueType.Float,
            default_value=1.0
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_convnext_projection_dim",
            values=[
                4, 8, 16, 32, 64, 128, 256
            ]
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_convnext_drop_path_rate",
            range=(0, 1),
            type=HyperparameterValueType.Float
        )
    )

    hp_manager.set_hp_format(
        HyperparameterFormat(
            name="hp_skip_connection",
            values=[
                "none", "concatenate"
            ]
        )
    )

    return hp_manager


__global_hp_manager = get_default_hp_manager()


def get_global_hp_manager():
    return __global_hp_manager


__all__ = [
    "HyperparametersManager", "HyperparameterValueType",
    "HyperparameterFormat", "get_global_hp_manager", "get_default_hp_manager"
]

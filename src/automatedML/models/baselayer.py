from .hpmanager import *
import random

_map_symbol_to_mode = {
    '!': "cascade_mode",
    "|": "parallel_mode",
    '&': "freeze_mode",
    '£': "partial_freeze_mode",
    '#': "skip_connection_mode"
}

_map_mode_to_symbol = {
    _map_symbol_to_mode[k]: k
    for k in _map_symbol_to_mode
}

_list_symbols = list(_map_symbol_to_mode.keys())


def _extract_modes_from_type(type: str):

    start_sub_type = type.find('(')
    end_sub_type = type.rfind(')')

    if start_sub_type >= 0 and end_sub_type >= 0:
        sub_type = type[start_sub_type+1:end_sub_type]
        flag_code = type.replace(f"({sub_type})", '')
    elif start_sub_type < 0 and end_sub_type < 0:
        flag_code = []
        sub_type = []
        ignore_mode = False
        for f in type:
            if not ignore_mode and f in _list_symbols:
                flag_code.append(f)
            else:
                ignore_mode = True
                sub_type.append(f)
        flag_code = ''.join(flag_code)
        sub_type = ''.join(sub_type)
    else:
        raise Exception("Not supported")

    obj = {_map_symbol_to_mode[f]: True for f in flag_code}
    return obj, sub_type


class BaseLayer():
    __hyperparameters: dict
    __hyperparameters_formats: dict[str, HyperparameterFormat]
    __flags: dict[str, bool]
    __disable_clone: bool

    @property
    def layerKey(self) -> str:
        s = [
            f"{x}={str(self.__hyperparameters[x])}"
            for x in self.__hyperparameters_formats
        ]
        if len(s) > 0:
            return f"{self.extendedType}-{'-'.join(s)}"
        else:
            return self.extendedType

    @property
    def extendedType(self):
        type_code = self.get_type()
        flag_code = ''.join([_map_mode_to_symbol[k]
                            for k in self.__flags if self.__flags[k]])
        code = flag_code+type_code

        s = [
            f"{x}={str(self.__hyperparameters[x])}"
            for x in self.__hyperparameters
            if not x in self.__hyperparameters_formats
        ]
        if len(s) > 0:
            return f"{code}:{':'.join(s)}"
        else:
            return code

    @property
    def metadata(self) -> list:
        return [self.extendedType] + self.get_hpvector()

    @property
    def count_hyperparameters(self):
        return len(self.__hyperparameters_formats)

    @property
    def has_hyperparameters(self):
        return len(self.__hyperparameters_formats) > 0

    @property
    def is_cascade(self):
        return self.__flags.get("cascade_mode", False)

    @property
    def is_parallel(self):
        return self.__flags.get("parallel_mode", False)

    @property
    def is_frozen(self):
        return self.__flags.get("freeze_mode", False)

    @property
    def is_skip_connection(self):
        return self.__flags.get("skip_connection_mode", False)

    def __init__(self, **kwargs) -> None:

        # set global data
        self.__hyperparameters = {}
        self.__hyperparameters_formats = {}
        self.__flags = {}
        self.__disable_clone = kwargs.pop("disable_clone", False)

        if not kwargs.pop("is_cloned", False):

            # check flags
            flags = {k: kwargs.pop(k, False)
                     for k in _map_mode_to_symbol.keys()}

            # register hyperparams
            hp_manager: HyperparametersManager = kwargs.pop("hp_manager",
                                                            get_global_hp_manager())
            self._register_custom_hyperparameters(hp_manager)

            # set additional hyperparameters
            if flags["cascade_mode"]:
                self._register_hyperparameter(
                    name="num_cascade_layer",
                    format=hp_manager.hp_num_cascade_layers
                )

            if flags["parallel_mode"]:
                self._register_hyperparameter(
                    name="num_parallel_layer",
                    format=hp_manager.hp_num_parallel_layer
                )
                self._register_hyperparameter(
                    name="merge_type",
                    format=hp_manager.hp_merge_types
                )

            if flags["skip_connection_mode"]:
                self._register_hyperparameter(
                    name="skip_connection",
                    format=hp_manager.hp_skip_connection
                )

            # update hyperparams
            if kwargs is not None and len(kwargs) > 0:
                updated_hps = self.update_hyperparameters(**kwargs)
                if flags["partial_freeze_mode"]:
                    for hpname in updated_hps:
                        del self.__hyperparameters_formats[hpname]

                for u in updated_hps:
                    kwargs.pop(u)
                self.__hyperparameters.update(kwargs)

            if flags["freeze_mode"]:
                self.__hyperparameters_formats.clear()

            # update flags
            self.__flags.update(flags)

    def __repr__(self):
        return self.layerKey

    # abstract
    def is_trainable(self) -> bool:
        raise Exception("Is Trainable not defined")

    # abstract
    def get_type(self) -> str:
        raise Exception("Type not defined")

    # abstract
    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        pass

    # abstract
    def _internal_compile(self, prev_layer, initializer):
        raise Exception("Internal Compile not defined")

    def _register_hyperparameter(self, name: str, format: HyperparameterFormat, default_value=None, readonly: bool = False):
        name = name.lower()
        assert not name in self.__hyperparameters.keys(
        ), f"{name} is already registered"
        self.__hyperparameters[name] = default_value if default_value is not None else format.default_value
        if not readonly:
            self.__hyperparameters_formats[name] = format

    def _custom_clone(self, target: 'BaseLayer'):
        pass

    def is_flag_enabled(self, flag: str) -> bool:
        return self.__flags.get(flag.lower(), False)

    def compile(self, prev_layer, initializer):
        from ..engine import get_layer_function

        # get local variables
        n_layers = self.__hyperparameters["num_cascade_layer"] if self.is_cascade else 1
        n_parallel = self.__hyperparameters["num_parallel_layer"] if self.is_parallel else 1
        skip_connection = self.__hyperparameters["skip_connection"] if self.is_skip_connection else "none"

        # compute parallel layers
        branch_list = []
        for _ in range(n_parallel):
            t = prev_layer
            for _ in range(n_layers):
                t = self._internal_compile(t, initializer)
            branch_list.append(t)

        # concate branches based on merge strategy
        if len(branch_list) > 1:
            merge_type = self.get_hyperparameter("merge_type")
            x = get_layer_function(merge_type)(branch_list)
        else:
            x = branch_list[0]

        # add skip connection
        if skip_connection != "none":
            # TODO check compatibile outout otherwise find a solutions
            x = get_layer_function(skip_connection)([x, prev_layer])
        return x

    def get_hyperparameter_format(self, name: str):
        return self.__hyperparameters_formats[name.lower()]

    def get_hyperparameter_formats(self) -> list[HyperparameterFormat]:
        if not self.is_frozen:
            return list(self.__hyperparameters_formats.values())
        else:
            return []

    def get_hyperparameter_bounds(self) -> list[list]:
        hp_lists = []
        if not self.is_frozen:
            for hp_format in self.__hyperparameters_formats.values():
                hp_lists.append(hp_format.get_discrete_values())
        return hp_lists

    def update_hyperparameters(self, **kwargs):
        l = []
        for key in self.__hyperparameters_formats.keys():
            val = kwargs.get(key, None)
            if val is not None:
                self.set_hyperparameter(key, val, ignore_constraint=True)
                l.append(key)
        return l

    def get_hyperparameter_names(self) -> list:
        return list(self.__hyperparameters_formats.keys())

    def get_hyperparameter_dic(self) -> dict:
        if not self.is_frozen:
            d = {
                name: self.__hyperparameters[name]
                for name in self.__hyperparameters_formats.keys()
            }
            return d
        else:
            return {}

    def get_hpvector(self) -> list:
        if not self.is_frozen:
            d = [
                self.__hyperparameters[name]
                for name in self.__hyperparameters_formats.keys()
            ]
            return d
        else:
            return []

    def get_hyperparameter(self, name: str, use_default=False, default=None):
        name = name.lower()

        if not use_default:
            v = self.__hyperparameters.get(name, None)
            if v is None:
                raise Exception(f"Hyperparameter {name} not supported")
        else:
            v = self.__hyperparameters.get(name, default)
        return v

    def set_hyperparameter(self, name: str, value, ignore_constraint: bool = False):
        assert not self.is_frozen, "The layer is frozen"

        name = name.lower()

        # check available type
        hp_format = self.__hyperparameters_formats.get(name, None)
        if hp_format is None:
            raise Exception(f"{name} is not registered")

        # check type
        value = hp_format.check_type(value)

        # check contraint
        if not ignore_constraint and not hp_format.check_constraints(value):
            raise Exception(f"{value} not valid in {name}")

        # set value
        self.__hyperparameters[name] = value

    def set_hpvector(self, hpvector: list, ignore_constraint: bool = False):
        assert not self.is_frozen, "The layer is frozen"

        if hpvector is None or len(self.__hyperparameters_formats) != len(hpvector):
            raise Exception("Format hpvector not supported")

        for i, name in enumerate(self.__hyperparameters_formats.keys()):
            self.set_hyperparameter(name, hpvector[i], ignore_constraint)

    def set_random_hyperparameters(self):
        assert not self.is_frozen, "The layer is frozen"
        for hp_name in self.__hyperparameters_formats.keys():
            values = self.__hyperparameters_formats[hp_name]
            self.__hyperparameters[hp_name] = values.get_random_value()

    def freeze(self, hpparams: list = None):
        if hpparams is None:
            self.__flags["freeze_mode"] = True
            self.__hyperparameters_formats.clear()
        else:
            if isinstance(hpparams, str):
                hpparams = [hpparams]

            self.__flags["partial_freeze_mode"] = True
            for hpname in hpparams:
                del self.__hyperparameters_formats[hpname]

    def clone(self):
        if self.__disable_clone:
            return self
        else:
            c: BaseLayer = self.__class__.__call__(is_cloned=True)
            for name in self.__hyperparameters.keys():
                c.__hyperparameters[name] = self.__hyperparameters[name]
            for name in self.__hyperparameters_formats.keys():
                c.__hyperparameters_formats[name] = self.__hyperparameters_formats[name]
            for name in self.__flags:
                c.__flags[name] = self.__flags[name]
            self._custom_clone(c)
            return c


class BaseSequentialBlock(BaseLayer):
    __layers: list[BaseLayer]

    def __init__(self, **kwargs):
        if not kwargs.get("is_cloned", False):

            # get hpmanager
            hp_manager: HyperparametersManager = kwargs.get("hp_manager",
                                                            get_global_hp_manager())

            # check layer
            layers = kwargs.pop("layers", None)
            if layers is None:
                layers = self.get_layers_in_block(hp_manager)
            if layers is None:
                raise Exception("Block configuration not valid")
            if not isinstance(layers, list):
                layers = [layers]
            elif len(layers) == 0:
                raise Exception("Block configuration not valid")

            # precompute layers
            self.__layers = []
            for layer in layers:
                if isinstance(layer, BaseLayer):
                    instance_layer = layer
                else:
                    instance_layer: BaseLayer = layer(hp_manager=hp_manager)
                self.__layers.append(instance_layer)

        super().__init__(**kwargs)

    def is_trainable(self) -> bool:
        for item in self.__layers:
            layer: BaseLayer = item
            if layer.is_trainable():
                return True
        return False

    def get_type_sublayers(self) -> list:
        return [s.extendedType for s in self.__layers]

    # abstract
    def get_layers_in_block(self, hp_manager: HyperparametersManager) -> list:
        raise Exception("Block Layers not defined")

    def _custom_clone(self, target: 'BaseSequentialBlock'):
        target.__layers = [x.clone() for x in self.__layers]

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        for layer_index, layer in enumerate(self.__layers):
            instance_layer: BaseLayer = layer
            for hp_name in instance_layer.get_hyperparameter_names():
                self._register_hyperparameter(
                    name=f"l{layer_index}{hp_name}",
                    format=instance_layer.get_hyperparameter_format(hp_name),
                    default_value=instance_layer.get_hyperparameter(hp_name)
                )
        pass

    def _internal_compile(self, prev_layer, initializer):

        x = prev_layer
        for layer_index, layer in enumerate(self.__layers):
            instance_layer: BaseLayer = layer

            for hp_name in instance_layer.get_hyperparameter_names():
                hp_value = self.get_hyperparameter(
                    f"L{layer_index}{hp_name}"
                )
                instance_layer.set_hyperparameter(hp_name, hp_value)

            x = instance_layer._internal_compile(x, initializer)
        return x


class BaseParallelBlock(BaseLayer):
    __layers: list[BaseLayer]

    def __init__(self, **kwargs) -> None:
        if not kwargs.get("is_cloned", False):

            # get hpmanager
            hp_manager: HyperparametersManager = kwargs.get("hp_manager",
                                                            get_global_hp_manager())

            # check layer
            layers = kwargs.pop("layers", None)
            if layers is None:
                layers = self.get_layer_in_parallel(hp_manager)
            if layers is None:
                raise Exception("Parallel configuration not valid")
            if not isinstance(layers, list):
                layers = [layers]
            elif len(layers) == 0:
                raise Exception("Parallel configuration not valid")

            # precompute layers
            self.__layers = []
            for layer in layers:
                if isinstance(layer, BaseLayer):
                    instance_layer = layer
                else:
                    instance_layer: BaseLayer = layer(hp_manager=hp_manager)
                self.__layers.append(instance_layer)

        super().__init__(**kwargs)

    def is_trainable(self) -> bool:
        for item in self.__layers:
            layer: BaseLayer = item
            if layer.is_trainable():
                return True
        return False

    def get_type_sublayers(self) -> list:
        return [s.extendedType for s in self.__layers]

    # abstract

    def get_layer_in_parallel(self, hp_manager: HyperparametersManager) -> list:
        raise Exception("Parallel Layers not defined")

    def _custom_clone(self, target: 'BaseParallelBlock'):
        target.__layers = [x.clone() for x in self.__layers]

    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("merge_type",
                                      hp_manager.hp_merge_types)

        for layer_index, layer in enumerate(self.__layers):
            instance_layer: BaseLayer = layer
            for hp_name in instance_layer.get_hyperparameter_names():
                self._register_hyperparameter(
                    name=f"b{layer_index}{hp_name}",
                    format=instance_layer.get_hyperparameter_format(hp_name),
                    default_value=instance_layer.get_hyperparameter(hp_name)
                )

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import get_layer_function

        merge_type = self.get_hyperparameter("merge_type")

        x = prev_layer
        y = []
        for layer_index, layer in enumerate(self.__layers):
            instance_layer: BaseLayer = layer

            for hp_name in instance_layer.get_hyperparameter_names():
                hp_value = self.get_hyperparameter(
                    f"b{layer_index}{hp_name}"
                )
                instance_layer.set_hyperparameter(hp_name, hp_value)

            y.append(instance_layer.compile(x, initializer))

        x = get_layer_function(merge_type)(y)
        return x


__all__ = ["BaseLayer", "BaseSequentialBlock",
           "BaseParallelBlock", "_extract_modes_from_type"]

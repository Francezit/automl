import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from .internal import TypeOfTask


class DataContainer():

    _X_train: np.ndarray
    _y_train: np.ndarray
    _X_test: np.ndarray
    _y_test: np.ndarray

    _type_of_task: TypeOfTask = TypeOfTask.REGRESSION
    _number_of_class: int = None
    _name: str = None
    _labels: list[str] = None
    _train_ticks: np.ndarray = None
    _test_ticks: np.ndarray = None

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                 type_of_task: TypeOfTask | str, number_of_class: int = None, name: str = "data", labels: list[str] = None, ticks: tuple[np.ndarray, np.ndarray] = None):

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        self._number_of_class = number_of_class
        self._labels = labels
        if ticks:
            if isinstance(ticks[0], list):
                self._train_ticks = np.array(ticks[0])
            else:
                self._train_ticks = ticks[0]

            if isinstance(ticks[1], list):
                self._test_ticks = np.array(ticks[1])
            else:
                self._test_ticks = ticks[1]

        if isinstance(type_of_task, TypeOfTask):
            self._type_of_task = type_of_task
        else:
            self._type_of_task = TypeOfTask(type_of_task)

        if name is None:
            self._name = "data"
        else:
            self._name = name

    def get_trainingset(self):
        return self._X_train, self._y_train

    def get_testset(self):
        return self._X_test, self._y_test

    @property
    def type_of_task(self):
        return self._type_of_task

    @property
    def number_of_class(self):
        return self._number_of_class

    @property
    def name(self):
        return self._name

    @property
    def labels(self):
        if self._labels is None:
            return [f"Var_{i}" for i in range(self.n_variables)]
        else:
            return self._labels

    @property
    def test_ticks(self):
        if self._test_ticks is None:
            n_train = self._X_train.shape[0]
            n_test = self._X_test.shape[0]
            return np.array([str(x) for x in range(n_train, n_train+n_test)])
        else:
            return self._test_ticks

    @property
    def ticks(self):
        return self.train_ticks, self.test_ticks

    @property
    def train_ticks(self):
        if self._train_ticks is None:
            n_train = self._X_train.shape[0]
            return np.array([str(x) for x in range(n_train)])
        else:
            return self._train_ticks

    @property
    def input_size(self) -> tuple:
        if len(self._X_train.shape) > 1:
            return self._X_train.shape[1:]
        else:
            return (1,)

    @property
    def output_size(self) -> tuple:
        if self._number_of_class is not None:
            return self._number_of_class
        elif len(self._y_train.shape) > 1:
            return self._y_train.shape[1:]
        else:
            return (1,)

    @property
    def n_variables(self):
        return self._X_train.shape[-1]

    @property
    def n_train_samples(self) -> int:
        return self._X_train.shape[0]

    @property
    def n_test_samples(self) -> int:
        return self._X_test.shape[0]

    @property
    def n_samples(self):
        return self.n_test_samples+self.n_train_samples


def init_data_container(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, type_of_task: TypeOfTask,
                        number_of_class: int = None, name: str = "data", labels: list[str] = None, ticks: tuple[np.ndarray, np.ndarray] = None,
                        standardization: str = None, use_balancing: bool = False, reduction_factor: float = 1, target_shape_size: int = None):

    if isinstance(type_of_task, str):
        type_of_task = TypeOfTask(type_of_task)

    if type_of_task.is_discrete_task:
        from .engine import to_categorical

        if number_of_class is None:
            raise Exception(
                "Number of class must be presented when type of task is classification")

        y_test = to_categorical(
            y_test,
            number_of_class
        )
        y_train = to_categorical(
            y_train,
            number_of_class
        )

    if standardization is not None:
        X_train, X_test = normalize_data(
            x_train=X_train,
            x_test=X_test,
            method=standardization
        )

    if use_balancing or (reduction_factor is not None and reduction_factor < 1):
        if isinstance(ticks, tuple):
            ticks = list(ticks)

        X_train, y_train, indices = downsample_data(
            x=X_train,
            y=y_train,
            reduction_factor=reduction_factor,
            type_of_task=type_of_task,
            balance_class=use_balancing,
            seed=1234
        )
        ticks[0] = ticks[0][indices]

        X_test, y_test, indices = downsample_data(
            x=X_test,
            y=y_test,
            reduction_factor=reduction_factor,
            type_of_task=type_of_task,
            balance_class=use_balancing,
            seed=1234
        )
        ticks[1] = ticks[1][indices]

    if target_shape_size is not None:
        while len(X_train.shape) < target_shape_size:
            n = len(X_train.shape)
            X_train = np.expand_dims(X_train, n)
            X_test = np.expand_dims(X_test, n)

    opt = DataContainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        ticks=ticks,
        name=name,
        number_of_class=number_of_class,
        type_of_task=type_of_task
    )

    return opt


def transform_data_container(opt: DataContainer, **kargs):
    return init_data_container(
        X_train=opt._X_train,
        y_train=opt._y_train,
        X_test=opt._X_test,
        y_test=opt._y_test,
        labels=opt._labels,
        name=opt._name,
        ticks=opt.ticks,
        number_of_class=opt._number_of_class,
        type_of_task=opt._type_of_task,
        **kargs
    )


def load_data_container(param: dict | str) -> DataContainer:
    data_container: DataContainer = None

    if isinstance(param, dict):
        from .engine import dataset_sources, load_dataset

        source = param.pop("source", 'keras')
        if isinstance(source, str) and source in dataset_sources:
            dataset_name = param["name"]
            X_train, y_train, X_test, y_test = load_dataset(
                source=source,
                dataset_name=dataset_name
            )
        elif isinstance(source, dict):
            def read_datafile(attr_name: str):
                f: str = source[attr_name]
                if os.path.exists(f):
                    if f.endswith(".npy"):
                        return np.load(f, allow_pickle=True)
                    else:
                        return pd.read_csv(f).to_numpy()
                else:
                    raise Exception(f"File not exist: {f}")

            X_train = read_datafile("trainX")
            y_train = read_datafile("trainY")
            X_test = read_datafile("testX")
            y_test = read_datafile("testY")
        else:
            raise Exception("DATA Not supported")

        data_container = DataContainer(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            labels=param.get("labels", None),
            number_of_class=param.get("labels", None),
            type_of_task=param["type_of_task"],
            ticks=param.get("ticks", None),
            name=param.get("name", None)
        )

    elif isinstance(param, str):
        import zipfile
        import json

        filename = param
        param = {}

        ticks = []
        dataset = []
        cell_names = ["X_train", "y_train", "X_test", "y_test"]
        with zipfile.ZipFile(filename, "r") as zfp:
            for attr_name in cell_names:
                with zfp.open(attr_name, "r") as fp:
                    dt = np.load(fp)
                    dataset.append(dt["arr_0"])

            for attr_name in ["train_ticks", "test_ticks"]:
                if attr_name in zfp.NameToInfo:
                    with zfp.open(attr_name, "r") as fp:
                        dt = np.load(fp)
                        ticks.append(dt["arr_0"])
                else:
                    ticks.append(None)

            with zfp.open("param", "r") as fp:
                txt: str = fp.read().decode()
                if txt != '':
                    param = json.loads(txt)
                else:
                    param = {}

        X_train, y_train, X_test, y_test = dataset

        data_container = DataContainer(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ticks=ticks,
            **param
        )
    else:
        raise Exception("Argument not supported")

    return data_container


def extract_data_container(filename: str, train_split: float, type_of_task: TypeOfTask | str, select_variables: list[str] | dict[str, str] = None,
                           convert_data_type: dict = None, name=None, index_type: str = None, index_format: str = None, filter_data_fun=None,
                           plot_stat_folder: str = None, file_sep: str = None, index_cols: list[str] = None,
                           use_first_row_as_header: bool = False, use_standardization: bool = False,
                           reduce_data_dimensionality: int = None, **kargs):

    # Import necessary libraries
    from sklearn import datasets  # to retrieve the iris Dataset
    # to standardize the features
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA  # to apply PCA
    import seaborn as sns  # to plot the heat maps

    if name is not None:
        name = os.path.basename(filename)
        name = name[:name.find('.')]

    # check variable type
    if not isinstance(type_of_task, TypeOfTask):
        type_of_task = TypeOfTask(type_of_task)

    # load dataset
    if not use_first_row_as_header:
        data = pd.read_csv(filename, header=None)
    else:
        data = pd.read_csv(filename, sep=file_sep)

    use_stat = plot_stat_folder

    # merge cols index
    if index_cols:
        target_name = ''.join(index_cols)
        if len(index_cols) > 1:
            data[target_name] = data[index_cols].agg(
                lambda x: " ".join([str(y) for y in x]),
                axis=1
            )
            data.drop(columns=index_cols, inplace=True)
        data.set_index(target_name, inplace=True)

    if index_type:
        if index_type == "datetime":
            index_col = data.index.name
            data.reset_index(inplace=True)
            data[index_col] = pd.to_datetime(
                data[index_col],
                format=index_format
            )
            data.set_index(index_col, inplace=True)
        pass

    if select_variables:
        if isinstance(select_variables, list) and len(select_variables) > 0:
            data = data[select_variables]
        elif isinstance(select_variables, dict):
            dt = data[list(select_variables.keys())]
            data = dt.rename(columns=select_variables)

    if convert_data_type:
        default_type = convert_data_type.pop("__default", None)
        if default_type is not None:
            for col in data.columns:
                if not col in convert_data_type:
                    convert_data_type[col] = default_type

        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.astype(convert_data_type)

    if filter_data_fun:
        data = filter_data_fun(filename, data)

    raw_ticks = data.index.to_numpy()
    labels = data.columns.to_list()

    if reduce_data_dimensionality:
        n_features = data.shape[1]
        n = None
        if isinstance(reduce_data_dimensionality, int):
            n = reduce_data_dimensionality
        elif isinstance(reduce_data_dimensionality, float):
            n = int(reduce_data_dimensionality*n_features)

        if n and n < n_features:
            pca = PCA(n_components=n)
            pca.fit(data)
            data_pca = pca.transform(data)
            data_pca = pd.DataFrame(data_pca)
            data = data_pca

    # handle statistic
    if use_stat:
        os.makedirs(plot_stat_folder, exist_ok=True)
        data.describe().to_csv(os.path.join(plot_stat_folder, "stat.csv"))

        label_names = data.columns.to_list()
        corr = data.corr()
        corr.to_csv(os.path.join(plot_stat_folder, "corr.csv"))

        fig = plt.figure()
        plt.matshow(corr, fignum=fig.number)
        plt.xlabel(label_names)
        plt.ylabel(label_names)
        plt.colorbar()
        fig.savefig(os.path.join(plot_stat_folder, "corr.pdf"))
        plt.close(fig)

    if use_standardization:
        scalar = MinMaxScaler()
        scaled_data = pd.DataFrame(
            scalar.fit_transform(data))  # scaling the data
        data = scaled_data

    # split dataset and build trainset and testset
    n_samples = data.shape[0]
    if type_of_task.is_forecasting:
        data.interpolate(method='linear', inplace=True, limit_direction="both")
        data = data.to_numpy()

        n_steps = kargs.pop("n_step", 1)
        n_target_steps = kargs.pop("n_target_steps", 1)

        x, y = list(), list()
        for i in range(n_samples):
            # find the end of this pattern
            end_ix = i + n_steps
            end_iy = end_ix+n_target_steps
            # check if we are beyond the sequence
            if end_iy > n_samples - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix, :], data[end_ix:end_iy, :]
            x.append(seq_x)
            y.append(seq_y)
        x, y = np.array(x), np.array(y)
        raw_ticks = raw_ticks[:x.shape[0]]

        n_train = int(np.floor(n_samples*train_split))
        container = init_data_container(
            X_train=x[:n_train, :],
            X_test=x[n_train:, :],
            y_train=y[:n_train, :],
            y_test=y[n_train:, :],
            type_of_task=type_of_task,
            ticks=(raw_ticks[:n_train], raw_ticks[n_train:]),
            name=name,
            labels=labels,
            number_of_class=None,
            **kargs
        )

        return container
    else:
        raise Exception("Type of Task not supported")


def save_data_container(opt: DataContainer, filename: str):
    import zipfile
    import json

    with zipfile.ZipFile(filename, "w") as zfp:

        def store(attr_name, d):
            with zfp.open(attr_name, "w") as fp:
                np.savez_compressed(fp, d)

        X_train, y_train = opt.get_trainingset()
        store("X_train", X_train)
        store("y_train", y_train)

        X_test, y_test = opt.get_testset()
        store("X_test", X_test)
        store("y_test", y_test)

        if opt._train_ticks is not None:
            store("train_ticks", opt._train_ticks.tolist())

        if opt._test_ticks is not None:
            store("test_ticks", opt._test_ticks.tolist())

        with zfp.open("param", "w") as fp:
            txt = json.dumps({
                "type_of_task": opt.type_of_task.value,
                "number_of_class": opt.number_of_class,
                "name": opt.name,
                "labels": opt.labels
            })
            fp.write(txt.encode())


def minmax_scale(a):
    a_min = a.min(axis=-2, keepdims=True)
    a_max = a.max(axis=-2, keepdims=True)
    out = (a - a_min) / (a_max - a_min)
    return out


def normalize_data(x_train: np.ndarray, x_test: np.ndarray, method: str):
    assert len(x_train.shape) == len(x_test.shape)
    from sklearn.preprocessing import scale

    if "divide" in method:
        f = int(method.split('_')[1])

        x_train = x_train.astype('float32') / f
        x_test = x_test.astype('float32') / f
        return x_train, x_test
    elif "min-max" in method:
        x_train = minmax_scale(x_train)
        x_test = minmax_scale(x_test)
        return x_train, x_test
    elif "scale" in method:
        x_train = scale(x_train)
        x_test = scale(x_test)
        return x_train, x_test
    else:
        raise Exception(f"{method} not valid")


def downsample_data(x: np.ndarray, y: np.ndarray, reduction_factor: float, type_of_task: TypeOfTask, seed: int = None, balance_class: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert x.shape[0] == y.shape[0]

    data_len = x.shape[0]

    if seed is None:
        shuffle = np.random.shuffle
    else:
        rdn = np.random.RandomState(seed)
        shuffle = rdn.shuffle

    if type_of_task.is_classification:
        indices = []
        y_code = np.argmax(y, axis=1)
        classes, counts = np.unique(y_code, return_counts=True)
        for class_code, count in zip(classes, counts):
            class_indices = np.where(y_code == class_code)[0]
            shuffle(class_indices)

            class_n = int(np.floor(count * reduction_factor))
            if balance_class:
                class_n = min(np.min(counts), class_n)

            indices.append(class_indices[:class_n])
        indices = np.concatenate(indices)
        return x[indices], y[indices], indices

    elif type_of_task.is_forecasting:
        n_samples = int(np.floor(data_len * reduction_factor))
        indices = np.array(range(data_len), dtype=np.int32)

        indices = indices[data_len-n_samples:]
        return x[indices], y[indices], indices

    else:
        n_samples = int(np.floor(data_len * reduction_factor))
        indices = np.array(range(data_len), dtype=np.int32)
        shuffle(indices)

        indices = indices[0:n_samples]
        return x[indices], y[indices], indices


def split_data(x: np.ndarray, y: np.ndarray, split: float):
    assert 0 < split < 1, "Split ratio must be between 0 and 1."

    n_samples = x.shape[0]
    n_train = int(np.floor(n_samples * (1 - split)))

    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    return x_train, y_train, x_val, y_val


__all__ = [
    "DataContainer", "init_data_container",
    "transform_data_container", "load_data_container", "extract_data_container",
    "save_data_container","split_data"
]

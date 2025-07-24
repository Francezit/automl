class TupleMatrix():
    _n_points: int
    _mem: dict
    _default_value: float

    @property
    def n_points(self):
        return self._n_points

    def __init__(self, n_points: int, default_value: float = 1) -> None:
        self._n_points = n_points
        self._mem = dict()
        self._default_value = default_value

    def __len__(self):
        return self._n_points

    def _check_key(self, item1: list, item2: list):
        if not isinstance(item1, tuple):
            item1 = tuple(item1)
        if not isinstance(item2, tuple):
            item2 = tuple(item2)
        return item1, item2

    def set(self, item1: tuple, item2: tuple, value: float):
        item1, item2 = self._check_key(item1, item2)

        row: dict = self._mem.get(item1, {})
        row[item2] = value
        self._mem[item1] = row

    def get(self, item1: tuple, item2: tuple):
        item1, item2 = self._check_key(item1, item2)

        row: dict = self._mem.get(item1, {})
        return row.get(item2, self._default_value)

    def reset(self):
        self._mem.clear()

    def rescale(self, factor: float):
        self._default_value *= factor
        for row in self._mem.values():
            for idx2 in row.keys():
                row[idx2] = row[idx2]*factor

    def mult(self, item1: tuple, item2: tuple, value: float):
        item1, item2 = self._check_key(item1, item2)

        row: dict = self._mem.get(item1, {})
        row[item2] = value * row.get(item2, self._default_value)
        self._mem[item1] = row

    def sum(self, item1: tuple, item2: tuple, value: float):
        item1, item2 = self._check_key(item1, item2)

        row: dict = self._mem.get(item1, {})
        row[item2] = value + row.get(item2, self._default_value)
        self._mem[item1] = row

    def operation(self, item1: tuple, item2: tuple, op):
        item1, item2 = self._check_key(item1, item2)

        row: dict = self._mem.get(item1, {})
        row[item2] = op(row.get(item2, self._default_value))
        self._mem[item1] = row


class BigMatrix():
    __n_points: int
    __mem: dict
    __default_value: float

    @property
    def shape(self):
        return (self.__n_points, self.__n_points)

    @property
    def n_points(self):
        return self.__n_points

    def __init__(self, n_points: int, default_value: float = 1) -> None:
        self.__n_points = n_points
        self.__mem = dict()
        self.__default_value = default_value

    def __len__(self):
        return self.__n_points*self.__n_points

    def set(self, idx1: int, idx2: int, value: float):
        assert idx1 >= 0 and idx2 >= 0
        assert idx1 < self.__n_points and idx2 < self.__n_points

        row: dict = self.__mem.get(idx1, {})
        row[idx2] = value
        self.__mem[idx1] = row

    def get(self, idx1: int, idx2: int):
        assert idx1 >= 0 and idx2 >= 0
        assert idx1 < self.__n_points and idx2 < self.__n_points

        row: dict = self.__mem.get(idx1, {})
        return row.get(idx2, self.__default_value)

    def reset(self):
        self.__mem.clear()

    def rescale(self, factor: float):
        self.__default_value *= factor
        for row in self.__mem.values():
            for idx2 in row.keys():
                row[idx2] = row[idx2]*factor

    def mult(self, idx1: int, idx2: int, value: float):
        assert idx1 >= 0 and idx2 >= 0
        assert idx1 < self.__n_points and idx2 < self.__n_points

        row: dict = self.__mem.get(idx1, {})
        row[idx2] = value * row.get(idx2, self.__default_value)
        self.__mem[idx1] = row

    def sum(self, idx1: int, idx2: int, value: float):
        assert idx1 >= 0 and idx2 >= 0
        assert idx1 < self.__n_points and idx2 < self.__n_points

        row: dict = self.__mem.get(idx1, {})
        row[idx2] = value + row.get(idx2, self.__default_value)
        self.__mem[idx1] = row

    def operation(self, idx1: int, idx2: int, op):
        assert idx1 >= 0 and idx2 >= 0
        assert idx1 < self.__n_points and idx2 < self.__n_points

        row: dict = self.__mem.get(idx1, {})
        row[idx2] = op(row.get(idx2, self.__default_value))
        self.__mem[idx1] = row


__all__=["TupleMatrix","BigMatrix"]
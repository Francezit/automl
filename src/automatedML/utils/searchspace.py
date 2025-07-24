import numpy as np
import random


class SearchSpace():
    _bounds: list
    _maximum_cache_size: int

    _curr_idx: int
    _cache: dict
    _len: int

    @property
    def bounds(self):
        return self._bounds

    @property
    def shape(self):
        return [len(x) for x in self._bounds]

    @property
    def is_iterable(self):
        return self._curr_idx < self._len

    def __init__(self, bounds: list, maximum_cache_size: int = None):
        self._bounds = bounds
        self._maximum_cache_size = maximum_cache_size

        self._curr_idx = 0
        self._cache = {}
        self._len = np.prod(self.shape)

    def __len__(self):
        return self._len

    def __next__(self) -> int:
        if self._curr_idx >= self._len:
            raise StopIteration()

        v = self.__getitem__(self._curr_idx)
        self._curr_idx += 1
        return v

    def __iter__(self):
        v = SearchSpace(self._bounds, self._maximum_cache_size)
        for k in self._cache.keys():
            v._cache[k] = self._cache[k].copy()
        return v

    def __getitem__(self, i: int) -> list:
        self._check()

        if i in self._cache:
            return self._cache[i]
        elif i == 0:
            v = [b[0] for b in self._bounds]
            self._cache[i] = v
            return v
        elif i > 0 and i < self._len:
            prev: list = self.__getitem__(i-1)
            v = prev.copy()
            self._increase(v)
            self._cache[i] = v
            return v
        else:
            raise Exception("Index out of range")

    def get_random_solutions(self, size: int = None):
        n = 1 if size is None else size
        assert n < self._len

        l = []
        while len(l) < n:
            v = []
            for b in self._bounds:
                v.append(b[random.randint(0, len(b)-1)])
            l.append(v)
        return l[0] if size is None else l

    def index(self, solution: list):
        pass

    def _increase(self, v: list, idx: int = None):
        if idx is None:
            idx = len(self._bounds)-1

        l: list = self._bounds[idx]
        lidx = l.index(v[idx])
        if lidx < len(l)-1:
            v[idx] = l[lidx+1]
        elif idx > 0:
            v[idx] = l[0]
            self._increase(v, idx-1)

    def _check(self):
        if self._maximum_cache_size is not None and len(self._cache) > self._maximum_cache_size*2:
            keys = list(self._cache.keys())
            keys.pop()
            random.shuffle(keys)
            keys = keys[len(keys) - self._maximum_cache_size:]
            for x in keys:
                self._cache.pop(x)


__all__ = ["SearchSpace"]

if __name__ == "__main__":
    import itertools

    bounds = [list(range(5)),
              list(range(8)),
              list(range(2)),
              list(range(8)),
              list(range(2)),
              list(range(4)),
              list(range(2)),
              list(range(2))]

    space = SearchSpace(bounds, 50)
    target_space = itertools.product(*bounds)

    l, t_l = [], []
    for t, s in zip(target_space, space):
        l.append(s)
        t_l.append(t)
        print(f"{t}->{s}")
        assert list(t) == s

    assert len(l) == len(space) and len(t_l) == len(space)
    print(len(space))

    el = space[10]
    print(el)

    solutions = space.get_random_solutions(10)

    pass

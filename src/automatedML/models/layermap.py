import numpy as np

from ..utils import SearchSpace


class LayerMap():
    __layer_map: dict[str, SearchSpace]

    @property
    def search_space_size(self) -> int:
        return int(np.prod([len(x) for x in self.__layer_map.values()]))

    @property
    def classes(self) -> list[str]:
        return list(self.__layer_map.keys())
    
    @property
    def n_classes(self) -> int: 
        return len(self.__layer_map)

    def __init__(self, layer_map: dict[str, list]) -> None:
        self.__layer_map = {}
        for k, v in layer_map.items():
            self.__layer_map[k.lower()] = SearchSpace(v)

    def __len__(self):
        return len(self.__layer_map)

    def at(self, index: int):
        k = list(self.__layer_map.keys())
        return self.__layer_map[k[index]]

    def get(self, class_name: str):
        return self.__layer_map[class_name.lower()]


__all__ = ["LayerMap"]

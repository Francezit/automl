import numpy as np

from .base import KnowledgeGraphInterface


class EmptyKnowledgeGraph(KnowledgeGraphInterface):
    _initNodeName = "Init"
    _metadataClassPosition = 0

    _class_names: list

    _classgraph: np.ndarray
    _classgraph_class2index: dict
    _classgraph_index2class: dict

    _itemgraph: np.ndarray
    _itemgraph_index2item: dict
    _itemgraph_item2index: dict

    _class_item_mapindex: dict
    _class_bestfitness_map: dict

    _is_loaded = False

    def __init__(self, layer_maps: dict) -> None:
        if layer_maps is not None:
            # init class graph. Add a fake node to set the start point
            class_names = [self._initNodeName] + list(layer_maps.keys())
            self._class_names = class_names
            n = len(class_names)
            self._classgraph = np.zeros((n, n))
            self._classgraph_class2index = dict(zip(class_names, range(0, n)))
            self._classgraph_index2class = dict(zip(range(0, n), class_names))

            items = [[self._initNodeName]] + [item for sublist in list(
                layer_maps.values()) for item in sublist]
            m = len(items)
            self._itemgraph = np.zeros((m, m))
            self._itemgraph_index2item = dict(zip(range(0, m), items))
            self._itemgraph_item2index = dict(
                zip([self._metadata2string(item) for item in items], range(0, m)))

            self._class_item_mapindex = dict()
            self._class_item_mapindex[0] = list(range(1, m))
            for class_index in range(1, n):
                class_name = self._classgraph_index2class[class_index]
                class_item_indexs = [self._itemgraph_item2index[self._metadata2string(
                    x)] for x in layer_maps[class_name]]
                self._class_item_mapindex[class_index] = class_item_indexs

            self._class_bestfitness_map = dict(
                zip(list(layer_maps.keys()), [np.inf for _ in range(len(layer_maps))]))

            self._is_loaded = True
            pass

    def _internal_clone(self, cloned):
        cloned._initNodeName = self._initNodeName
        cloned._metadataClassPosition = self._metadataClassPosition

        cloned._class_names = self._class_names.copy()

        cloned._classgraph = self._classgraph.copy()
        cloned._classgraph_class2index = self._classgraph_class2index.copy()
        cloned._classgraph_index2class = self._classgraph_index2class.copy()

        cloned._itemgraph = self._itemgraph.copy()
        cloned._itemgraph_index2item = self._itemgraph_index2item.copy()
        cloned._itemgraph_item2index = self._itemgraph_item2index.copy()

        cloned._class_item_mapindex = self._class_item_mapindex.copy()
        cloned._class_bestfitness_map = self._class_bestfitness_map.copy()
        pass

    def clone(self):
        cloned = EmptyKnowledgeGraph(None)
        self._internal_clone(cloned)
        return cloned

    def choice_next_layer(self, prev_metadata_sequence: list):
        # Select a random layer without using the knowledge

        availableClassIndexs = list(range(1, len(self._class_names)))
        n_class_names = len(availableClassIndexs)
        p = 1/(n_class_names)
        availableClassProbabilities = [p for _ in range(n_class_names)]
        selectedClassIndex = int(np.random.choice(
            availableClassIndexs, 1, p=availableClassProbabilities))

        availableItemIndexs = self._class_item_mapindex[selectedClassIndex]
        n_available_items = len(availableItemIndexs)
        p = 1/n_available_items
        availableItemProbabilities = [p for _ in range(n_available_items)]
        selectedItemIndex = int(np.random.choice(
            availableItemIndexs, 1, p=availableItemProbabilities))

        layer = self._itemgraph_index2item[selectedItemIndex]
        return layer

    def _metadata2string(self, metadata: list):
        return '#'.join([str(x) for x in metadata])

    def _string2metadata(self, string: str):
        return string.split('#')

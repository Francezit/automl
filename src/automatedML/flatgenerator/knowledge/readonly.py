import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

from .empty import EmptyKnowledgeGraph

# TODO calcolare le statistiche


class ReadOnlyKnowledgeGraph(EmptyKnowledgeGraph):

    _history: list = []

    @property
    def number_evaluation(self):
        return len(self._history)

    def get_evaluation_history(self):
        return [x.copy() for x in self._history]

    def get_stat(self):
        return {
            "number_evaluation": len(self._history),
            "number_hyperparameter_tested": np.count_nonzero(self._itemgraph)
        }

    def __init__(self, layer_maps: dict):
        super().__init__(layer_maps)

    def _internal_clone(self, cloned):
        cloned = ReadOnlyKnowledgeGraph(None)
        super()._internal_clone(cloned)

        cloned._history = [x.copy() for x in self._history]

    def clone(self):
        cloned = ReadOnlyKnowledgeGraph(None)
        self._internal_clone(cloned)
        return cloned

    def save(self, filename: str):
        with open(filename, "w") as fp:
            json.dump({
                "version": 1,
                "classgraph": [x.tolist() for x in self._classgraph],
                "classgraph_labels": list(self._classgraph_class2index.keys()),
                "history":  self._history,
                "itemgraph": [x.tolist() for x in self._itemgraph],
                "itemgraph_labels": list(self._itemgraph_item2index.keys()),
            }, fp)

    def plot(self, extended_version: bool = False, map_node_labels: dict = None, show_edge_labels: bool = False, normalize_weights: bool = False):
        if map_node_labels is None:
            map_node_labels = {}

        fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})

        class_names = [map_node_labels.get(x, x) for x in self._class_names]
        color_map = dict(zip(class_names,
                             list(range(len(class_names)))))

        classgraph = self._classgraph
        if normalize_weights:
            classgraph_min = np.abs(classgraph.min())+1
            for r in range(classgraph.shape[0]):
                for c in range(classgraph.shape[1]):
                    if classgraph[r, c] != 0:
                        classgraph[r, c] = classgraph[r, c] + classgraph_min
            classgraph = classgraph/classgraph.max()

        itemgraph = self._itemgraph
        if normalize_weights:
            itemgraph_min = np.abs(itemgraph.min())+1
            for r in range(itemgraph.shape[0]):
                for c in range(itemgraph.shape[1]):
                    if itemgraph[r, c] != 0:
                        itemgraph[r, c] = itemgraph[r, c] + itemgraph_min
            itemgraph = itemgraph/itemgraph.max()

        if not extended_version:
            G = nx.MultiDiGraph()
            for name in self._classgraph_class2index.keys():
                G.add_node(map_node_labels.get(name, name))

            for idx1 in range(classgraph.shape[0]):
                for idx2 in range(classgraph.shape[1]):
                    score = classgraph[idx1, idx2]
                    if score != 0:
                        u_name = map_node_labels.get(self._classgraph_index2class[idx1],
                                                     self._classgraph_index2class[idx1])
                        v_name = map_node_labels.get(self._classgraph_index2class[idx2],
                                                     self._classgraph_index2class[idx2])
                        G.add_edge(u_name,
                                   v_name,
                                   value=score)
            node_colors = [color_map[node] for node in G.nodes]
        else:
            G = nx.MultiDiGraph()

            for idx1 in range(itemgraph.shape[0]):
                for idx2 in range(itemgraph.shape[1]):
                    score = itemgraph[idx1, idx2]
                    if score != 0:
                        u_item = self._itemgraph_index2item[idx1].copy()
                        u_item[0] = map_node_labels.get(u_item[0], u_item[0])

                        v_item = self._itemgraph_index2item[idx2].copy()
                        v_item[0] = map_node_labels.get(v_item[0], v_item[0])

                        G.add_edge(u_for_edge=self._metadata2string(u_item),
                                   v_for_edge=self._metadata2string(v_item),
                                   value=score)
            node_colors = [
                color_map[self._string2metadata(node)[self._metadataClassPosition]] for node in G]

        edge_labels = dict(
            [((edge[0], edge[1]), f'{edge[2]["value"]}') for edge in G.edges(data=True)])

        edge_color = np.array([edge[2]["value"]
                              for edge in G.edges(data=True)])
        # edge_color = (edge_color-min(edge_color)) / \
        #    (max(edge_color)-min(edge_color))

        pos = nx.circular_layout(G)
        # nx.draw_networkx_nodes(G,pos=pos,node_color=node_colors)
        # nx.draw_networkx_labels(G,pos=pos,font_size=6)
        # maxedges=nx.draw_networkx_edges(G,pos=pos,arrows=True,
        #                       edge_color=edge_color,edge_cmap=plt.cm.Blues,
        #                       connectionstyle='arc3,rad=0.1')

        edge_cmap = mpl.cm.RdYlGn
        node_cmap = mpl.cm.gist_rainbow
        nx.draw(G, ax=axes[0], pos=pos, node_color=node_colors, font_size=6,
                with_labels=True, arrows=True, width=0.7, edge_color=edge_color,
                connectionstyle='arc3,rad=0.1', cmap=node_cmap, edge_cmap=edge_cmap)

        if show_edge_labels:
            nx.draw_networkx_edge_labels(G, pos=pos,
                                         edge_labels=edge_labels, font_size=6)

        norm = mpl.colors.Normalize(vmin=min(edge_color), vmax=max(edge_color))
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=edge_cmap),
                     cax=axes[1], orientation='vertical')

        return G, fig

    def load(self, filename: str) -> bool:

        data: dict = None
        with open(filename, "r") as fp:
            data = json.load(fp)

        version = data.get("version", 0)
        if version < 1:
            return False

        # merge the data
        classgraph = np.array([np.array(x) for x in data["classgraph"]])
        classgraph_labels: list = data["classgraph_labels"]
        if self._is_loaded:
            for class_index_1 in range(classgraph.shape[0]):
                for class_index_2 in range(classgraph.shape[1]):
                    class_name_1 = classgraph_labels[class_index_1]
                    class_name_2 = classgraph_labels[class_index_2]
                    value = classgraph[class_index_1, class_index_2]

                    new_index_1 = self._classgraph_class2index.get(
                        class_name_1, None)
                    new_index_2 = self._classgraph_class2index.get(
                        class_name_2, None)
                    if new_index_1 is not None and new_index_2 is not None:
                        self._classgraph[new_index_1, new_index_2] = value
        else:
            self._class_names = classgraph_labels
            self._classgraph = classgraph
            self._classgraph_class2index = dict(zip(classgraph_labels,
                                                    list(range(0, len(classgraph_labels)))))
            self._classgraph_index2class = dict(zip(list(range(0, len(classgraph_labels))),
                                                    classgraph_labels))
        del classgraph, classgraph_labels

        # class_bestfitness_map: dict = data["class_bestfitness_map"]
        # for class_name in class_bestfitness_map.keys():
        #    self._class_bestfitness_map[class_name] = class_bestfitness_map[class_name]

        itemgraph = np.array([np.array(x)for x in data["itemgraph"]])
        itemgraph_labels: dict = data["itemgraph_labels"]
        if self._is_loaded:
            for item_index_1 in range(itemgraph.shape[0]):
                for item_index_2 in range(itemgraph.shape[1]):
                    item_1 = itemgraph_labels[item_index_1]
                    item_2 = itemgraph_labels[item_index_2]
                    value = itemgraph[item_index_1, item_index_2]

                    new_index_1 = self._itemgraph_item2index.get(
                        item_1, None)
                    new_index_2 = self._itemgraph_item2index.get(
                        item_2, None)
                    if new_index_1 is not None and new_index_2 is not None:
                        self._itemgraph[new_index_1, new_index_2] = value
        else:
            self._itemgraph = itemgraph
            self._itemgraph_item2index = dict(zip(itemgraph_labels,
                                                  list(range(0, len(itemgraph_labels)))))
            self._itemgraph_index2item = dict(zip(list(range(0, len(itemgraph_labels))),
                                                  [self._string2metadata(item) for item in itemgraph_labels]))
        del itemgraph, itemgraph_labels

        history: list = data.get("history", [])
        if self._is_loaded:
            for item in history:
                self._history.append(item)
        else:
            self._history = history
        del history

        return True

    def choice_next_layer(self, prev_metadata_sequence: list):
        # each item is a tuple containing the class name of layer and a list of hyperparameter
        # given a sequence of base layer, this code return a new layer
        # I take into account only the last layer at the moment

        if len(prev_metadata_sequence) > 0:
            prev_metadata_sequence = prev_metadata_sequence[-1]
            currentClassName = prev_metadata_sequence[self._metadataClassPosition]
        else:
            prev_metadata_sequence = [self._initNodeName]
            currentClassName = self._initNodeName

        # choice the next class
        currentClassIndex = self._classgraph_class2index[currentClassName]
        availableClassIndexs = list(range(1, len(self._class_names)))
        availableClassScores = self._classgraph[currentClassIndex,
                                                availableClassIndexs]
        availableClassProbabilities = self._compute_probabilities(
            scores=availableClassScores
        )
        selectedClassIndex = int(np.random.choice(a=availableClassIndexs,
                                                  size=1,
                                                  p=availableClassProbabilities))

        # select the hyperparameters
        currentItem = self._metadata2string(prev_metadata_sequence)
        currentItemIndex = self._itemgraph_item2index[currentItem]
        availableItemIndexs = self._class_item_mapindex[selectedClassIndex]
        availableItemScores = self._itemgraph[currentItemIndex,
                                              availableItemIndexs]
        availableItemProbabilities = self._compute_probabilities(
            scores=availableItemScores
        )
        selectedItemIndex = int(np.random.choice(a=availableItemIndexs,
                                                 size=1,
                                                 p=availableItemProbabilities))

        layer = self._itemgraph_index2item[selectedItemIndex]
        return layer

    def _compute_probabilities(self, scores: np.ndarray):
        if len(scores.shape) == 2 and scores.shape[1] > 1:
            probs = np.zeros(shape=scores.shape)
            for i in range(scores.shape[0]):
                row = scores[i, :]
                probs[i, :] = np.exp2(row) / np.sum(np.exp2(row))
            return probs
        else:
            return np.exp2(scores) / np.sum(np.exp2(scores))

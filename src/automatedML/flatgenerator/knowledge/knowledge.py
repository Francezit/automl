import numpy as np

from .readonly import ReadOnlyKnowledgeGraph

class KnowledgeGraphRewardOptions():
    positive_reward: float = 1
    opt_reward: float = 0.5
    stable_reward: float = 0
    negative_reward: float = -1
    invalid_reward: float = negative_reward*10

    def __init__(self, obj: dict = None):
        if obj is not None:
            for key in obj.keys():
                self.__setattr__(key, obj[key])


class KnowledgeGraph(ReadOnlyKnowledgeGraph):

    _positive_reward: float = 1
    _opt_reward: float = 0.5
    _stable_reward: float = 0
    _negative_reward: float = -1
    _invalid_reward: float = _negative_reward*10

    def __init__(self, layer_maps: dict, reward_options:  KnowledgeGraphRewardOptions = None):
        super().__init__(layer_maps)
        if reward_options is not None:
            if isinstance(reward_options, dict):
                reward_options = KnowledgeGraphRewardOptions(reward_options)

            self._positive_reward = reward_options.positive_reward
            self._opt_reward = reward_options.opt_reward
            self._stable_reward = reward_options.stable_reward
            self._negative_reward = reward_options.negative_reward
            self._invalid_reward = reward_options.invalid_reward
        pass

    def clone(self):
        cloned = KnowledgeGraph(None)
        self._internal_clone(cloned)

        cloned._positive_reward = self._positive_reward
        cloned._stable_reward = self._stable_reward
        cloned._negative_reward = self._negative_reward
        cloned._invalid_reward = self._invalid_reward
        cloned._opt_reward = self._opt_reward

        return cloned

    def register(self, metadata_sequence: list, fitness: list):

        # check size
        if len(metadata_sequence) != len(fitness):
            raise Exception("Arguments not valid")

        # handle not valid first layers
        fist_valid_layer_index: int = None
        for i in range(len(metadata_sequence)):
            if np.isinf(fitness[i]):
                self._change_weight_graph(metadata1=[self._initNodeName],
                                          metadata2=metadata_sequence[i],
                                          value=self._invalid_reward)
                self._history.append({
                    "layers": [metadata_sequence[i]],
                    "cost": fitness[i]
                })
            else:
                fist_valid_layer_index = i
                break

        # check if there is at least one valid layer
        if fist_valid_layer_index is not None:

            # handle first valid layer
            firstClassName = metadata_sequence[fist_valid_layer_index][self._metadataClassPosition]
            bestFitness = min(fitness)
            currentBestFitness = self._class_bestfitness_map[firstClassName]
            if bestFitness < currentBestFitness:
                firstClassReward = self._positive_reward
                self._class_bestfitness_map[firstClassName] = bestFitness
            elif bestFitness == currentBestFitness:
                firstClassReward = self._stable_reward
            else:
                firstClassReward = self._negative_reward

            self._change_weight_graph(metadata1=[self._initNodeName],
                                      metadata2=metadata_sequence[fist_valid_layer_index],
                                      value=firstClassReward)

            # update reward in the next layers
            currentSeq = metadata_sequence[fist_valid_layer_index]
            currentFitness = fitness[fist_valid_layer_index]
            for i in range(fist_valid_layer_index+1, len(metadata_sequence)):
                nextFitness = fitness[i]
                nextSeq = metadata_sequence[i]

                if np.isinf(nextFitness):  # it is not valid
                    self._change_weight_graph(metadata1=currentSeq,
                                              metadata2=nextSeq,
                                              value=self._invalid_reward)
                else:
                    reward: float
                    if nextFitness < currentFitness:  # it is improving
                        reward = self._positive_reward
                    elif nextFitness == currentFitness:  # it is stable
                        reward = self._stable_reward
                    else:  # it is decreasing
                        reward = self._negative_reward

                    self._change_weight_graph(metadata1=currentSeq,
                                              metadata2=nextSeq,
                                              value=reward)

                    currentSeq = nextSeq
                    currentFitness = nextFitness
                pass

            # update history
            layerHistory = []
            for i in range(fist_valid_layer_index, len(metadata_sequence)):
                self._history.append({
                    "layers": layerHistory.copy()+[metadata_sequence[i]],
                    "cost": fitness[i]
                })
                if np.isfinite(fitness[i]):
                    layerHistory.append(metadata_sequence[i])
            del layerHistory
        pass

    def update(self, metadata_sequence: list, fitness: float):

        # handle first layer
        firstClassName = metadata_sequence[0][self._metadataClassPosition]
        currentBestFitness = self._class_bestfitness_map[firstClassName]
        if fitness < currentBestFitness:
            firstClassReward = self._positive_reward
            self._class_bestfitness_map[firstClassName] = fitness
        elif fitness == currentBestFitness:
            firstClassReward = self._stable_reward
        else:
            firstClassReward = self._negative_reward

        self._change_weight_graph(metadata1=[self._initNodeName],
                                  metadata2=metadata_sequence[0],
                                  value=firstClassReward)

        # handle other layers
        currentSeq = metadata_sequence[0]
        for i in range(1, len(metadata_sequence)):
            nextSeq = metadata_sequence[i]
            self._change_weight_graph(metadata1=currentSeq,
                                      metadata2=nextSeq,
                                      value=self._opt_reward)

            currentSeq = nextSeq

        self._history.append({
            "layers": metadata_sequence,
            "cost": fitness
        })
        pass

    def _change_weight_graph(self, metadata1: list, metadata2: list, value: int):
        itemidx1 = self._itemgraph_item2index[self._metadata2string(metadata1)]
        itemidx2 = self._itemgraph_item2index[self._metadata2string(metadata2)]
        self._itemgraph[itemidx1,
                        itemidx2] = self._itemgraph[itemidx1, itemidx2]+value

        classidx1 = self._classgraph_class2index[metadata1[self._metadataClassPosition]]
        classidx2 = self._classgraph_class2index[metadata2[self._metadataClassPosition]]
        self._classgraph[classidx1,
                         classidx2] = self._classgraph[classidx1, classidx2]+value
        pass

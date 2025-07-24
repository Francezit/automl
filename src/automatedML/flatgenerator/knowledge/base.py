class KnowledgeGraphInterface():
    def clone(self):
        raise Exception("Not Implemented")

    def save(self, filename: str):
        pass

    def plot(self, extended_version: bool = False, labels_map: dict = None, show_edge_labels: bool = False):
        return None

    def saveplot(self, filename: str, **kwargs):
        import matplotlib.pyplot as plt

        _, fig = self.plot(**kwargs)
        fig.savefig(filename)
        plt.close(fig)

    def load(self, filename: str) -> bool:
        return False

    def optimize(self):
        pass #TODO trovare un modo per ottimizzare i pesi degli archi dopo tanti episodi. Vedere problema saturazione ???

    def register(self, metadata_sequence: list, fitness: list):
        pass

    def update(self, metadata_sequence: list, fitness: float):
        pass

    def choice_next_layer(self, prev_metadata_sequence: list):
        raise Exception("Not Implemented")

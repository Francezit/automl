import numpy as np
from ..internal import ProblemType


class MetricInfo:
    loss: float
    density: float
    cost: float
    deepIndex: float
    other_metrics: list

    def __init__(self, cost: float, loss: float,  density: float, deepIndex: float, other_metrics: list = []) -> None:
        self.loss = loss
        self.density = density
        self.cost = cost
        self.deepIndex = deepIndex
        self.other_metrics = other_metrics

    @property
    def is_valid(self) -> bool:
        return bool(np.isfinite(self.cost))

    @property
    def is_empty(self) -> bool:
        return bool(np.isinf(self.cost))

    def to_dict(self, prefix: str = "") -> dict:
        if prefix is None:
            prefix = ""
        assert isinstance(prefix, str)

        return {
            f"{prefix}loss": self.loss,
            f"{prefix}density": self.density,
            f"{prefix}cost": self.cost,
            f"{prefix}deepIndex": self.deepIndex,
            f"{prefix}other_metrics": self.other_metrics,
        }

    def save(self, filename: str):
        import json
        with open(filename, "w") as fp:
            json.dump(self.to_dict(), fp)

    def summary(self, print_fn=None):
        if print_fn is None:
            print_fn = print

        obj = self.to_dict()
        for item in obj.keys():
            print_fn(f"{item}: {obj[item]}")

    def __str__(self) -> str:
        if self.loss is not None:
            return f"{self.cost}({';'.join([str(f) for f in self.other_metrics])})"
        else:
            return "inf"

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __ne__(self, other):
        return self.cost != other.cost

    @staticmethod
    def empty():
        return MetricInfo(np.inf, None, np.inf, np.inf)


def select_best_metric(metrics: list, field_name: str, type: ProblemType) -> int:
    assert len(metrics) > 0 and all(
        [isinstance(x, MetricInfo) for x in metrics]
    )

    v = [getattr(x, field_name) for x in metrics]
    if isinstance(v[0], list):
        v = [x[0] if len(x) > 0 else None for x in v]

    invalid_value = np.inf if type == ProblemType.MIN else -np.inf
    v = [x if x is not None and np.isfinite(x) else invalid_value for x in v]

    if type == ProblemType.MAX:
        return np.argmax(v)
    elif type == ProblemType.MIN:
        return np.argmin(v)
    else:
        raise Exception("Not supported type")


__all__=["MetricInfo","select_best_metric"]
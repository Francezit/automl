import pandas as pd

from .metricInfo import MetricInfo



class StatisticInfo():
    history: list
    filename: str
    autosave: bool

    def __init__(self, filename: str, autosave: bool = True) -> None:
        self.filename = filename
        self.autosave = autosave
        self.history = []

    def append(self, initMetric: MetricInfo, optMetric: MetricInfo, episodeMetric: MetricInfo, bestMetric: MetricInfo,
               optimizeSuccess: bool, episode_time: float, fullMetric: MetricInfo):

        record: dict = {
            "episode": len(self.history),
            "optimizeSuccess": optimizeSuccess,
            "time": episode_time
        }
        record.update(initMetric.to_dict("init-"))
        record.update(optMetric.to_dict("opt-"))
        record.update(episodeMetric.to_dict("episode-"))
        record.update(bestMetric.to_dict("best-"))
        record.update(fullMetric.to_dict("real-"))
        self.history.append(record)

        if self.autosave:
            self.save()

    def save(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.filename, index=False)


__all__=["StatisticInfo"]
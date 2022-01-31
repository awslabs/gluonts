import os
import pickle
from pathlib import Path
from typing import Dict, List
from tsbench.config import Config, EnsembleConfig
from tsbench.evaluations.metrics import Performance
from ._base import Tracker
from ._evaluations import Evaluations


class EnsembleTracker(Tracker[EnsembleConfig]):
    """
    Tracker which sources information from .pickle files providing configurations of ensemble
    members and the ensembles' performances.
    """

    def __init__(self, directory: Path):
        """
        Args:
            files: Directory from which to load the files non-recursively.
        """
        configurations = []
        performances = []
        for file in os.listdir(directory):
            if not file.endswith(".pickle"):
                continue
            with Path(file).open("rb") as f:
                data = pickle.load(f)
                configurations.extend(
                    [
                        Config(frozenset(x["configurations"]), x["dataset"])
                        for x in data
                    ]
                )
                performances.extend([x["performance"] for x in data])

        self.performance_map: Dict[Config[EnsembleConfig], Performance] = dict(
            zip(configurations, performances)
        )

    def unique_ensembles(self) -> List[EnsembleConfig]:
        """
        Returns the unique configurations of ensembles provided by this tracker.
        """
        return list({c.model for c in self.performance_map})

    def get_evaluations(self) -> Evaluations[EnsembleConfig]:
        return Evaluations(
            list(self.performance_map),
            list(self.performance_map.values()),
        )

    def get_performance(self, config: Config[EnsembleConfig]) -> Performance:
        return self.performance_map[config]

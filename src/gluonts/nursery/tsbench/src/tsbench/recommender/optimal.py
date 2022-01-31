from typing import List, Optional
from tsbench.config import Config, ModelConfig
from tsbench.evaluations.metrics import Performance
from tsbench.evaluations.tracking import ModelTracker
from ._base import Recommender
from ._factory import register_recommender
from .generator import CandidateGenerator


@register_recommender("optimal")
class OptimalRecommender(Recommender[ModelConfig]):
    """
    The optimal recommender makes recommendations by accessing the true performances of the
    models.
    """

    def __init__(
        self,
        tracker: ModelTracker,
        objectives: List[str],
        focus: Optional[str] = None,
        generator: Optional[CandidateGenerator[ModelConfig]] = None,
    ):
        """
        Args:
            tracker: The tracker to obtain true performance metrics from.
            objectives: The list of performance metrics to minimize.
            focus: The metric to prefer. Must be either in the list of objectives. If not
                provided, the first metric to optimize is chosen.
            generator: The generator that generates configurations for recommendations. By default,
                this is the replay candidate generator.
        """
        super().__init__(objectives, focus, generator=generator)
        self.tracker = tracker

    def _get_performances(self, configs: List[Config[ModelConfig]]) -> List[Performance]:
        return [self.tracker.get_performance(c) for c in configs]

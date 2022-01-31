from typing import cast, List, Optional
from mxnet.gluon import nn
from mxnet.gluon.trainer import Trainer
from .base import Callback


class LearningRateScheduleCallback(Callback):  # type: ignore
    """
    The learning rate schedule callback decreases the learning rate by a predefined factor after
    each of the provided milestones (after x seconds during training).
    """

    def __init__(
        self,
        milestones: List[float],
        decay: float = 0.5,
    ):
        """
        Args:
            decay: The factor which to use for reducing the learning rate.
            milestones: The number of seconds after which the learning rate should be decreased
                according to the provided decay.
        """
        assert all(
            x < y for x, y in zip(milestones, milestones[1:])
        ), "Milestones must be increasing."

        super().__init__()
        self.lr = 0
        self.decay = decay
        self.milestones = milestones
        self.milestone_index = 0
        self.trainer: Optional[Trainer] = None

    def on_train_start(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.lr = trainer.learning_rate
        self.milestone_index = 0

    def on_train_batch_end(
        self, network: nn.HybridBlock, time_elapsed: float
    ) -> None:
        if (
            len(self.milestones) > self.milestone_index
            and time_elapsed > self.milestones[self.milestone_index]
        ):
            self.milestone_index += 1
            self.lr = self.lr * self.decay
            cast(Trainer, self.trainer).set_learning_rate(self.lr)

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, Generic, List, TypeVar, Union
import pandas as pd
from .dataset import DatasetConfig
from .model import ModelConfig, TrainConfig

EnsembleConfig = FrozenSet[ModelConfig]

T = TypeVar("T", ModelConfig, EnsembleConfig)


@dataclass(frozen=True)
class Config(Generic[T]):
    """
    A configuration is a tuple of a model configuration and a dataset configuration.
    """

    @classmethod
    def to_dataframe(
        cls,
        configs: List[Config[ModelConfig]],
    ) -> pd.DataFrame:
        """
        Returns a data frame representing the provided configurations. The model is translated into
        columns of the data frame automatically. The dataset is converted to a single column by
        being represented by its name.
        """
        rows: List[Dict[str, Union[str, float, int, bool]]] = [
            {
                "model": c.model.name(),
                "dataset": c.dataset.name(),
                **{
                    (
                        f"model_{k}"
                        if k in TrainConfig.training_hyperparameters()
                        else f"model_{c.model.name()}_{k}"
                    ): v
                    for k, v in asdict(c.model).items()
                },
            }
            for c in configs
        ]
        return pd.DataFrame(rows)

    model: T
    dataset: DatasetConfig

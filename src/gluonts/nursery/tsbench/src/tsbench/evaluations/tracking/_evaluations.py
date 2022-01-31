from dataclasses import dataclass
from typing import cast, Generic, List, TypeVar
import pandas as pd
from tsbench.config import Config, EnsembleConfig, ModelConfig
from tsbench.evaluations.metrics import Performance

T = TypeVar("T", ModelConfig, EnsembleConfig)


@dataclass
class Evaluations(Generic[T]):
    """
    Generic base for evaluations of models.
    """

    configurations: List[Config[T]]
    performances: List[Performance]

    def dataframe(self, std: bool = True) -> pd.DataFrame:
        """
        Returns a dataframe which contains the performance metrics as columns and the
        configurations as multi-index.

        Args:
            std: Whether to include the standard deviation of performance metrics in the dataframe.
        """
        # Should implement this for ensembles as well
        index_df = Config.to_dataframe(
            cast(List[Config[ModelConfig]], self.configurations)
        )
        # Reorder columns
        column_order = ["dataset"] + [
            c for c in index_df.columns.tolist() if c != "dataset"
        ]
        index = pd.MultiIndex.from_frame(index_df[column_order])
        df = Performance.to_dataframe(self.performances, std=std)
        df.index = index
        return df.sort_index()

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Predictor
from gluonts.dataset.util import period_index
from gluonts.dataset.split import split
from gluonts.dataset.util import forecast_start


def _to_dataframe(input_label: Tuple[DataEntry, DataEntry]) -> pd.DataFrame:
    """
    Turn a pair of consecutive (in time) data entries into a dataframe.
    """
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)


def make_oracle_predictions(
    dataset: Dataset,
    predictor: Predictor,
    num_samples: int = 100,
) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    """
    !!! CAN ONLY BE USED WITH ORACLE STYLE PREDICTORS !!!
    Oracle predictors are predictors that can access the future values of the
    target during prediction time. This function is used to evaluate such
    predictors by providing them with the actual future values of the target
    during prediction time.

    Parameters
    ----------
    dataset
        Dataset where the evaluation will happen. Only the portion excluding
        the prediction_length portion is used when making prediction.
    predictor
        Model used to draw predictions.
    num_samples
        Number of samples to draw on the model when evaluating. Only
        sampling-based models will use this.

    Returns
    -------
    Tuple[Iterator[Forecast], Iterator[pd.Series]]
        A pair of iterators, the first one yielding the forecasts, and the
        second one yielding the corresponding ground truth series.
    """

    window_length = predictor.prediction_length + getattr(
        predictor, "lead_time", 0
    )
    _, test_template = split(dataset, offset=-window_length)
    test_data = test_template.generate_instances(window_length)

    return (
        predictor.predict(
            test_data.input,
            num_samples=num_samples,
            ground_truth=test_data.label,
        ),
        map(_to_dataframe, test_data),
    )


class TrueOraclePredictor(Predictor):
    @validated()
    def __init__(self, prediction_length: int, num_samples: int) -> None:
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        ground_truth = kwargs.pop("ground_truth", None)
        if ground_truth is None:
            raise ValueError(
                "Ground truth is required for TrueOraclePredictor"
            )
        for item, label in zip(dataset, ground_truth):
            yield self.predict_item(
                item, ground_truth=label[FieldName.TARGET], **kwargs
            )

    def predict_item(
        self, item: DataEntry, num_samples: int, ground_truth=None
    ) -> Forecast:
        forecast_start_time = forecast_start(item)
        samples = np.tile(
            ground_truth[-self.prediction_length :], (num_samples, 1)
        )
        return SampleForecast(
            samples=samples,
            start_date=forecast_start_time,
            item_id=item.get("item_id", None),
        )


class OffsetOraclePredictor(Predictor):
    @validated()
    def __init__(self, prediction_length: int, num_samples: int) -> None:
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        ground_truth = kwargs.pop("ground_truth", None)
        if ground_truth is None:
            raise ValueError(
                "Ground truth is required for OffsetOraclePredictor"
            )
        for item, label in zip(dataset, ground_truth):
            yield self.predict_item(
                item, ground_truth=label[FieldName.TARGET], **kwargs
            )

    def predict_item(
        self,
        item: DataEntry,
        num_samples: int,
        ground_truth=None,
        offset: int = 1,
    ) -> Forecast:
        forecast_start_time = forecast_start(item)
        offset_ground_truth = np.roll(
            ground_truth[-self.prediction_length :], shift=offset
        )
        samples = np.tile(offset_ground_truth, (num_samples, 1))
        return SampleForecast(
            samples=samples,
            start_date=forecast_start_time,
            item_id=item.get("item_id", None),
        )

from typing import Dict

import numpy as np
import pandas as pd

from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName


def make_predictions(
    pred_input: Dataset, predictor, forecasts_at_all_levels: bool = False
):
    predictor.prediction_net.return_forecasts_at_all_levels = \
        forecasts_at_all_levels
    forecast_it = predictor.predict(pred_input)
    return forecast_it


def truncate_target(dataset: Dataset, prediction_length: int, lead_time: int = 0):

    def _truncate(data: DataEntry):
        data = data.copy()
        target = data["target"]
        assert (
                target.shape[-1] >= prediction_length
        )  # handles multivariate case (target_dim, history_length)
        data["target"] = target[..., : -prediction_length - lead_time]
        return data

    return map(_truncate, dataset)


def to_dataframe_it(test):
    def period_index(entry: DataEntry, freq=None) -> pd.DatetimeIndex:
        if freq is None:
            freq = entry[FieldName.START].freq

        return pd.period_range(
            start=entry[FieldName.START],
            periods=entry[FieldName.TARGET].shape[-1],
            freq=freq,
        )

    def _to_dataframe(entry: Dict):
        return pd.DataFrame(entry["target"], index=period_index(entry))

    return map(_to_dataframe, test)


def crps(forecast, actuals, levels: np.ndarray = (np.arange(10) / 10.0)[1:], weighted: bool = True):
    quantiles = np.array(
        [
            forecast.quantile(level)
            for level in levels
        ]
    )

    levels = np.expand_dims(levels, axis=-1)

    crps = np.where(
        actuals >= quantiles,
        levels * (actuals - quantiles),
        (1 - levels) * (quantiles - actuals),
    )

    if weighted:
        return crps.sum() / actuals.sum()
    else:
        return crps.sum()

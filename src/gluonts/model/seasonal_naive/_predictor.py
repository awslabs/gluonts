# Standard library imports
from typing import Iterator, Optional

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.evaluation import get_seasonality
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor


class SeasonalNaivePredictor(RepresentablePredictor):
    """
    Seasonal naïve forecaster.

    For each time series :math:`y`, this predictor produces a forecast
    :math:`\\tilde{y}(T+k) = y(T+k-h)`, where :math:`T` is the forecast time,
    :math:`k = 0, ...,` `prediction_length - 1`, and :math:`h =`
    `season_length`.

    If `prediction_length > season_length`, then the season is repeated
    multiple times. If a time series is shorter than season_length, then the
    mean observed value is used as prediction.

    Parameters
    ----------
    freq
        Frequency of the input data
    prediction_length
        Number of time points to predict
    season_length
        Length of the seasonality pattern of the input data
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        season_length: Optional[int] = None,
    ) -> None:
        super().__init__(prediction_length, freq)

        assert (
            season_length is None or season_length > 0
        ), "The value of `season_length` should be > 0"

        self.freq = freq
        self.prediction_length = prediction_length
        self.season_length = (
            season_length
            if season_length is not None
            else get_seasonality(freq)
        )

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[SampleForecast]:
        for data in dataset:
            start = pd.Timestamp(data['start'], freq=self.freq)
            target = np.asarray(data['target'], np.float32)
            yield self._predict_time_series(start_time=start, target=target)

    def _predict_time_series(
        self, start_time: pd.Timestamp, target: np.ndarray
    ) -> SampleForecast:
        len_ts = len(target)
        assert (
            len_ts >= 1
        ), "all time series should have at least one data point"

        if len_ts >= self.season_length:
            indices = [
                len_ts - self.season_length + k % self.season_length
                for k in range(self.prediction_length)
            ]
            samples = target[indices].reshape((1, self.prediction_length))
        else:
            samples = np.full(
                shape=(1, self.prediction_length), fill_value=target.mean()
            )

        forecast_time = start_time + len_ts * start_time.freq
        return SampleForecast(samples, forecast_time, start_time.freqstr)

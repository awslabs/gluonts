# Standard library imports
from typing import Callable, Dict, Iterator, Optional

# Third-party imports
import pandas as pd

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor

USAGE_MESSAGE = """
The `ProphetPredictor` is a thin wrapper for calling the Prophet package.
In order to use it you need to install fbprophet

pip install fbprophet

"""


class ProphetPredictor(RepresentablePredictor):
    """
    Wrapper around `Prophet <https://github.com/facebook/prophet>`_.

    The `ProphetPredictor` is a thin wrapper for calling the Prophet package.
    In order to use it you need to install fbprophet::

        pip install fbprophet

    Parameters
    ----------
    prediction_length
        Number of time points to predict
    freq
        Time frequency of the data, e.g. "1H"
    num_samples
        Number of samples to draw for predictions
    params
        Parameters to pass when instantiating the prophet model,
        e.g. `fbprophet.Prophet(**params)`
    model_callback
        An optional function that will be called with the configured model.
        This can be used to configure more complex setups, e.g.

    Examples
    --------
    >>> def configure_model(model):
    ...     model.add_seasonality(
    ...         name='weekly', period=7, fourier_order=3, prior_scale=0.1
    ...     )
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        num_eval_samples: int = 100,
        params: Optional[Dict] = None,
        model_callback: Optional[Callable] = None,
    ) -> None:
        try:
            import fbprophet
        except ImportError as e:
            raise ImportError(str(e) + USAGE_MESSAGE) from e

        self._fbprophet = fbprophet

        self.prediction_length = prediction_length
        self.freq = freq
        self.num_eval_samples = num_eval_samples
        self.params = params if params is not None else {}
        assert (
            'uncertainty_samples' not in self.params
        ), "parameter 'uncertainty_samples' should not be set directly. Please use num_samples."
        self._cb = (
            model_callback if model_callback is not None else lambda m: m
        )

        self.params['uncertainty_samples'] = self.num_eval_samples
        self.model_callback = model_callback

    def _prepare_input_df(self, d):
        index = pd.date_range(
            d["start"], periods=len(d["target"]), freq=self.freq
        )
        df = pd.DataFrame({"ds": index, "y": d["target"]})
        return df

    def predict(
        self, dataset: Dataset, num_eval_samples=None, **kwargs
    ) -> Iterator[SampleForecast]:
        for entry in dataset:
            if isinstance(entry, dict):
                data = entry
            else:
                data = entry.data
            params = self.params.copy()
            num_eval_samples = (
                num_eval_samples
                if num_eval_samples is not None
                else self.num_eval_samples
            )
            params['uncertainty_samples'] = num_eval_samples
            forecast = self._run_prophet(data, params)
            samples = forecast['yhat'].T
            forecast_start = pd.Timestamp(data['start'], freq=self.freq) + len(
                data['target']
            )
            assert samples.shape == (
                num_eval_samples,
                self.prediction_length,
            ), samples.shape
            yield SampleForecast(
                samples, forecast_start, forecast_start.freqstr
            )

    def _run_prophet(self, d, params):
        m = self._fbprophet.Prophet(**params)
        self._cb(m)
        inp = self._prepare_input_df(d)
        model = m.fit(inp)
        future_df = model.make_future_dataframe(
            self.prediction_length, freq=self.freq, include_history=False
        )
        forecast = model.predictive_samples(future_df)
        return forecast

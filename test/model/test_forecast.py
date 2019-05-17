# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.model.forecast import QuantileForecast, SampleForecast

SAMPLES = np.arange(0, 101).reshape(-1, 1) / 100.0
QUANTILES = SAMPLES[1:-1, 0]
START_DATE = pd.Timestamp(2017, 1, 1, 12)
FREQ = '1D'

FORECASTS = {
    'QuantileForecast': QuantileForecast(
        forecast_arrays=QUANTILES.reshape(-1, 1),
        start_date=START_DATE,
        forecast_keys=QUANTILES.tolist(),
        freq=FREQ,
    ),
    'SampleForecast': SampleForecast(
        samples=SAMPLES.reshape(len(SAMPLES), 1),
        start_date=START_DATE,
        freq=FREQ,
    ),
}


@pytest.mark.parametrize("fcst_cls", FORECASTS.keys())
def test_Forecast(fcst_cls):
    fcst = FORECASTS[fcst_cls]
    num_samples, pred_length = SAMPLES.shape

    # quantiles = [x/float(num_samples-1) for x in range(0, num_samples)]

    for q_value in QUANTILES:
        q_str = str(q_value)
        quantile_str = 'p{:02d}'.format(int(round(q_value * 100)))
        for q in [q_value, q_str, quantile_str]:
            quant_pred = fcst.quantile(q)
            assert (
                np.abs(quant_pred - q_value).reshape((1,)) < 1e-6
            ), "Expected {} quantile {}. Obtained {}.".format(
                q_value, q_value, quant_pred
            )
    assert fcst.prediction_length == 1
    assert len(fcst.index) == pred_length
    assert fcst.index[0] == pd.Timestamp(START_DATE)

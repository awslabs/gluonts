import numpy as np
import pandas as pd
import pytest
from itertools import islice
from gluonts.model.trivial.validation import (
    make_oracle_predictions,
    TrueOraclePredictor,
    OffsetOraclePredictor,
)
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import SampleForecast


@pytest.fixture
def dataset():
    return ListDataset(
        [{"start": "2020-01-01", "target": np.random.normal(size=100)}],
        freq="D",
    )


@pytest.fixture
def true_oracle_predictor():
    return TrueOraclePredictor(prediction_length=10, num_samples=5)


@pytest.fixture
def offset_oracle_predictor():
    return OffsetOraclePredictor(prediction_length=10, num_samples=5)


def test_make_oracle_predictions_true_oracle(dataset, true_oracle_predictor):
    forecasts, ground_truths = make_oracle_predictions(
        dataset, true_oracle_predictor
    )
    forecasts = list(islice(forecasts, 1))
    ground_truths = list(islice(ground_truths, 1))

    assert len(forecasts) == 1
    assert len(ground_truths) == 1

    forecast = forecasts[0]
    ground_truth = ground_truths[0]

    assert isinstance(forecast, SampleForecast)
    assert isinstance(ground_truth, pd.DataFrame)

    assert forecast.samples.shape == (5, 10)
    assert forecast.samples.mean() == pytest.approx(
        ground_truth.values[-10:].mean(), rel=1e-2
    )


def test_make_oracle_predictions_offset_oracle(
    dataset, offset_oracle_predictor
):
    forecasts, ground_truths = make_oracle_predictions(
        dataset, offset_oracle_predictor
    )
    forecasts = list(islice(forecasts, 1))
    ground_truths = list(islice(ground_truths, 1))

    assert len(forecasts) == 1
    assert len(ground_truths) == 1

    forecast = forecasts[0]
    ground_truth = ground_truths[0]

    assert isinstance(forecast, SampleForecast)
    assert isinstance(ground_truth, pd.DataFrame)

    assert forecast.samples.shape == (5, 10)
    assert forecast.samples.mean() == pytest.approx(
        np.roll(ground_truth.values[-10:], shift=1).mean(), rel=1e-2
    )

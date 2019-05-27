# Standard library imports
from random import randint

# Third-party imports
import pandas as pd
import pytest

# First-party imports
from gluonts.model.deepar import DeepAREstimator
from gluonts.transform import FieldName

NUM_TS = 5
START = "2018-01-01"
FREQ = "D"
TS_LENGTH = 20


def make_dummy_dataset_with_cat(num_ts, start, freq, ts_length, cardinality):
    return [
        {
            FieldName.START: pd.Timestamp(start, freq=freq),
            FieldName.TARGET: [1.0] * ts_length,
            FieldName.FEAT_STATIC_CAT: [randint(0, c) for c in cardinality],
        }
        if cardinality is not None
        else {
            FieldName.START: pd.Timestamp(start, freq=freq),
            FieldName.TARGET: [1.0] * ts_length,
        }
        for _ in range(num_ts)
    ]


@pytest.mark.parametrize(
    "estimator_type, hps",
    [
        (
            DeepAREstimator,
            dict(freq="D", prediction_length=7, num_batches_per_epoch=1),
        )
    ],
)
def test_no_cat(estimator_type, hps):

    dataset = make_dummy_dataset_with_cat(
        num_ts=NUM_TS,
        start=START,
        freq=FREQ,
        ts_length=TS_LENGTH,
        cardinality=None,
    )

    assert len(dataset) == NUM_TS
    assert FieldName.FEAT_STATIC_CAT not in dataset[0].keys()

    estimator = estimator_type.from_hyperparameters(**hps)
    predictor = estimator.train(dataset)
    forecasts = predictor.predict(dataset)

    assert len(list(forecasts)) == NUM_TS


@pytest.mark.parametrize(
    "estimator_type, hps",
    [
        (
            DeepAREstimator,
            dict(
                freq="D",
                prediction_length=7,
                num_batches_per_epoch=1,
                cardinality=[5],
            ),
        )
    ],
)
def test_one_cat(estimator_type, hps):

    dataset = make_dummy_dataset_with_cat(
        num_ts=NUM_TS,
        start=START,
        freq=FREQ,
        ts_length=TS_LENGTH,
        cardinality=[5],
    )

    assert len(dataset) == NUM_TS
    assert len(dataset[0][FieldName.FEAT_STATIC_CAT]) == 1

    estimator = estimator_type.from_hyperparameters(**hps)
    predictor = estimator.train(dataset)
    forecasts = predictor.predict(dataset)

    assert len(list(forecasts)) == NUM_TS


@pytest.mark.parametrize(
    "estimator_type, hps",
    [
        (
            DeepAREstimator,
            dict(
                freq="D",
                prediction_length=7,
                num_batches_per_epoch=1,
                cardinality=[5, 3, 10, 72],
            ),
        )
    ],
)
def test_multi_cat(estimator_type, hps):

    dataset = make_dummy_dataset_with_cat(
        num_ts=NUM_TS,
        start=START,
        freq=FREQ,
        ts_length=TS_LENGTH,
        cardinality=[5, 3, 10, 72],
    )

    assert len(dataset) == NUM_TS
    assert len(dataset[0][FieldName.FEAT_STATIC_CAT]) == 4

    estimator = estimator_type.from_hyperparameters(**hps)
    predictor = estimator.train(dataset)
    forecasts = predictor.predict(dataset)

    assert len(list(forecasts)) == NUM_TS

# Standard library imports
from random import randint
from typing import List, Tuple

# Third-party imports
import pytest

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.transform import FieldName
from gluonts.trainer import Trainer

NUM_TS = 5
START = "2018-01-01"
FREQ = "D"
MIN_LENGTH = 20
MAX_LENGTH = 50
PRED_LENGTH = 7


def _make_dummy_datasets_with_features(
    num_ts: int,
    start: str,
    freq: str,
    min_length: int,
    max_length: int,
    prediction_length: int = 0,
    cardinality: List[int] = [],
    num_feat_dynamic_real: int = 0,
) -> Tuple[ListDataset, ListDataset]:

    data_iter_train = []
    data_iter_test = []

    for k in range(num_ts):
        ts_length = randint(min_length, max_length)
        data_entry_train = {
            FieldName.START: start,
            FieldName.TARGET: [0.0] * ts_length,
        }
        if len(cardinality) > 0:
            data_entry_train[FieldName.FEAT_STATIC_CAT] = [
                randint(0, c) for c in cardinality
            ]
        data_entry_test = data_entry_train.copy()
        if num_feat_dynamic_real > 0:
            data_entry_train[FieldName.FEAT_DYNAMIC_REAL] = [
                [float(1 + k)] * ts_length
                for k in range(num_feat_dynamic_real)
            ]
            data_entry_test[FieldName.FEAT_DYNAMIC_REAL] = [
                [float(1 + k)] * (ts_length + prediction_length)
                for k in range(num_feat_dynamic_real)
            ]
        data_iter_train.append(data_entry_train)
        data_iter_test.append(data_entry_test)

    return (
        ListDataset(data_iter=data_iter_train, freq=freq),
        ListDataset(data_iter=data_iter_test, freq=freq),
    )


@pytest.mark.parametrize(
    "estimator, datasets",
    [
        # No features
        (
            DeepAREstimator(
                freq="D",
                prediction_length=PRED_LENGTH,
                trainer=Trainer(epochs=3, num_batches_per_epoch=2),
            ),
            _make_dummy_datasets_with_features(
                num_ts=NUM_TS,
                start=START,
                freq=FREQ,
                prediction_length=PRED_LENGTH,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            ),
        ),
        # Single static categorical feature
        (
            DeepAREstimator(
                freq="D",
                prediction_length=PRED_LENGTH,
                trainer=Trainer(epochs=3, num_batches_per_epoch=2),
                cardinality=[3],
            ),
            _make_dummy_datasets_with_features(
                num_ts=NUM_TS,
                start=START,
                freq=FREQ,
                prediction_length=PRED_LENGTH,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                cardinality=[3],
            ),
        ),
        # Multiple static categorical features
        (
            DeepAREstimator(
                freq="D",
                prediction_length=PRED_LENGTH,
                trainer=Trainer(epochs=3, num_batches_per_epoch=2),
                cardinality=[3, 10, 42],
            ),
            _make_dummy_datasets_with_features(
                num_ts=NUM_TS,
                start=START,
                freq=FREQ,
                prediction_length=PRED_LENGTH,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                cardinality=[3, 10, 42],
            ),
        ),
        # Single dynamic real feature
        (
            DeepAREstimator(
                freq="D",
                prediction_length=PRED_LENGTH,
                trainer=Trainer(epochs=3, num_batches_per_epoch=2),
            ),
            _make_dummy_datasets_with_features(
                num_ts=NUM_TS,
                start=START,
                freq=FREQ,
                prediction_length=PRED_LENGTH,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                num_feat_dynamic_real=1,
            ),
        ),
        # Multiple dynamic real feature
        (
            DeepAREstimator(
                freq="D",
                prediction_length=PRED_LENGTH,
                trainer=Trainer(epochs=3, num_batches_per_epoch=2),
            ),
            _make_dummy_datasets_with_features(
                num_ts=NUM_TS,
                start=START,
                freq=FREQ,
                prediction_length=PRED_LENGTH,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                num_feat_dynamic_real=5,
            ),
        ),
        # Both static categorical and dynamic real features
        (
            DeepAREstimator(
                freq="D",
                prediction_length=PRED_LENGTH,
                trainer=Trainer(epochs=3, num_batches_per_epoch=2),
                cardinality=[3, 10, 42],
            ),
            _make_dummy_datasets_with_features(
                num_ts=NUM_TS,
                start=START,
                freq=FREQ,
                prediction_length=PRED_LENGTH,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                cardinality=[3, 10, 42],
                num_feat_dynamic_real=5,
            ),
        ),
    ],
)
def test_deepar_smoke(estimator, datasets):
    dataset_train, dataset_test = datasets
    predictor = estimator.train(dataset_train)
    forecasts = predictor.predict(dataset_test)
    assert len(list(forecasts)) == len(dataset_test)

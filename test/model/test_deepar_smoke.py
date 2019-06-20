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


def _make_dummy_datasets_with_features(
    num_ts: int = 5,
    start: str = "2018-01-01",
    freq: str = "D",
    min_length: int = 5,
    max_length: int = 10,
    prediction_length: int = 3,
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


common_estimator_hps = dict(
    freq="D",
    prediction_length=3,
    trainer=Trainer(epochs=3, num_batches_per_epoch=2, batch_size=4),
)


@pytest.mark.parametrize(
    "estimator, datasets",
    [
        # No features
        (
            DeepAREstimator(**common_estimator_hps),
            _make_dummy_datasets_with_features(),
        ),
        # Single static categorical feature
        (
            DeepAREstimator(
                **common_estimator_hps,
                use_feat_static_cat=True,
                cardinality=[5],
            ),
            _make_dummy_datasets_with_features(cardinality=[5]),
        ),
        # Multiple static categorical features
        (
            DeepAREstimator(
                **common_estimator_hps,
                use_feat_static_cat=True,
                cardinality=[3, 10, 42],
            ),
            _make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        ),
        # Multiple static categorical features (ignored)
        (
            DeepAREstimator(**common_estimator_hps),
            _make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        ),
        # Single dynamic real feature
        (
            DeepAREstimator(
                **common_estimator_hps, use_feat_dynamic_real=True
            ),
            _make_dummy_datasets_with_features(num_feat_dynamic_real=1),
        ),
        # Multiple dynamic real feature
        (
            DeepAREstimator(
                **common_estimator_hps, use_feat_dynamic_real=True
            ),
            _make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        ),
        # Multiple dynamic real feature (ignored)
        (
            DeepAREstimator(**common_estimator_hps),
            _make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        ),
        # Both static categorical and dynamic real features
        (
            DeepAREstimator(
                **common_estimator_hps,
                use_feat_dynamic_real=True,
                use_feat_static_cat=True,
                cardinality=[3, 10, 42],
            ),
            _make_dummy_datasets_with_features(
                cardinality=[3, 10, 42], num_feat_dynamic_real=3
            ),
        ),
        # Both static categorical and dynamic real features (ignored)
        (
            DeepAREstimator(**common_estimator_hps),
            _make_dummy_datasets_with_features(
                cardinality=[3, 10, 42], num_feat_dynamic_real=3
            ),
        ),
    ],
)
def test_deepar_smoke(estimator, datasets):
    dataset_train, dataset_test = datasets
    predictor = estimator.train(dataset_train)
    forecasts = predictor.predict(dataset_test)
    assert len(list(forecasts)) == len(dataset_test)

# Standard library imports
from typing import Tuple

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
import gluonts
from gluonts import time_feature, transform
from gluonts.core import fqname_for
from gluonts.core.serde import dump_code, dump_json, load_code, load_json
from gluonts.dataset.common import ProcessStartField
from gluonts.dataset.stat import ScaleHistogram, calculate_dataset_statistics

FREQ = '1D'

TEST_VALUES = {
    'is_train': [True, False],
    'target': [np.zeros(0), np.random.rand(13), np.random.rand(100)],
    'start': [
        ProcessStartField.process('2012-01-02', freq='1D'),
        ProcessStartField.process('1994-02-19 20:01:02', freq='3D'),
    ],
    'use_prediction_features': [True, False],
    'allow_target_padding': [True, False],
}


def test_align_timestamp():
    def aligned_with(date_str, freq):
        return str(ProcessStartField.process(date_str, freq=freq))

    for _ in range(2):
        assert (
            aligned_with('2012-03-05 09:13:12', 'min') == '2012-03-05 09:13:00'
        )
        assert (
            aligned_with('2012-03-05 09:13:12', '2min')
            == '2012-03-05 09:12:00'
        )
        assert (
            aligned_with('2012-03-05 09:13:12', 'H') == '2012-03-05 09:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:13:12', 'D') == '2012-03-05 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:13:12', 'W') == '2012-03-04 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:13:12', '4W') == '2012-03-04 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:13:12', 'M') == '2012-02-29 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:13:12', '3M') == '2012-02-29 00:00:00'
        )

        assert (
            aligned_with('2012-03-05 09:14:11', 'min') == '2012-03-05 09:14:00'
        )
        assert (
            aligned_with('2012-03-05 09:14:11', '2min')
            == '2012-03-05 09:14:00'
        )
        assert (
            aligned_with('2012-03-05 09:14:11', 'H') == '2012-03-05 09:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:14:11', 'D') == '2012-03-05 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:14:11', 'W') == '2012-03-04 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:14:11', '4W') == '2012-03-04 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:14:11', 'M') == '2012-02-29 00:00:00'
        )
        assert (
            aligned_with('2012-03-05 09:14:11', '3M') == '2012-02-29 00:00:00'
        )


@pytest.mark.parametrize("is_train", TEST_VALUES['is_train'])
@pytest.mark.parametrize("target", TEST_VALUES['target'])
@pytest.mark.parametrize("start", TEST_VALUES['start'])
def test_AddTimeFeatures(start, target, is_train):
    pred_length = 13
    t = transform.AddTimeFeatures(
        start_field=transform.FieldName.START,
        target_field=transform.FieldName.TARGET,
        output_field='myout',
        pred_length=pred_length,
        time_features=[time_feature.DayOfWeek(), time_feature.DayOfMonth()],
    )

    assert_serializable(t)

    data = {'start': start, 'target': target}
    res = t.map_transform(data, is_train=is_train)
    mat = res['myout']
    expected_length = len(target) + (0 if is_train else pred_length)
    assert mat.shape == (2, expected_length)
    tmp_idx = pd.date_range(
        start=start, freq=start.freq, periods=expected_length
    )
    assert np.alltrue(mat[0] == time_feature.DayOfWeek()(tmp_idx))
    assert np.alltrue(mat[1] == time_feature.DayOfMonth()(tmp_idx))


@pytest.mark.parametrize("is_train", TEST_VALUES['is_train'])
@pytest.mark.parametrize("target", TEST_VALUES['target'])
@pytest.mark.parametrize("start", TEST_VALUES['start'])
def test_AddAgeFeatures(start, target, is_train):
    pred_length = 13
    t = transform.AddAgeFeature(
        pred_length=pred_length,
        target_field=transform.FieldName.TARGET,
        output_field='age',
        log_scale=True,
    )

    assert_serializable(t)

    data = {'start': start, 'target': target}
    out = t.map_transform(data, is_train=is_train)
    expected_length = len(target) + (0 if is_train else pred_length)
    assert out['age'].shape[-1] == expected_length
    assert np.allclose(
        out['age'],
        np.log10(2.0 + np.arange(expected_length)).reshape(
            (1, expected_length)
        ),
    )


@pytest.mark.parametrize("is_train", TEST_VALUES['is_train'])
@pytest.mark.parametrize("target", TEST_VALUES['target'])
@pytest.mark.parametrize("start", TEST_VALUES['start'])
def test_InstanceSplitter(start, target, is_train):
    train_length = 100
    pred_length = 13
    t = transform.InstanceSplitter(
        target_field=transform.FieldName.TARGET,
        is_pad_field=transform.FieldName.IS_PAD,
        start_field=transform.FieldName.START,
        forecast_start_field=transform.FieldName.FORECAST_START,
        train_sampler=transform.UniformSplitSampler(p=1.0),
        past_length=train_length,
        future_length=pred_length,
        time_series_fields=['some_time_feature'],
        pick_incomplete=True,
    )

    assert_serializable(t)

    other_feat = np.arange(len(target) + 100)
    data = {
        'start': start,
        'target': target,
        'some_time_feature': other_feat,
        'some_other_col': 'ABC',
    }

    out = list(t.flatmap_transform(data, is_train=is_train))

    if is_train:
        assert len(out) == max(0, len(target) - pred_length + 1)
    else:
        assert len(out) == 1

    for o in out:
        assert 'target' not in o
        assert 'some_time_feature' not in o
        assert 'some_other_col' in o

        assert len(o['past_some_time_feature']) == train_length
        assert len(o['past_target']) == train_length

        if is_train:
            assert len(o['future_target']) == pred_length
            assert len(o['future_some_time_feature']) == pred_length
        else:
            assert len(o['future_target']) == 0
            assert len(o['future_some_time_feature']) == pred_length

    # expected_length = len(target) + (0 if is_train else pred_length)
    # assert len(out['age']) == expected_length
    # assert np.alltrue(out['age'] == np.log10(2.0 + np.arange(expected_length)))


@pytest.mark.parametrize("is_train", TEST_VALUES['is_train'])
@pytest.mark.parametrize("target", TEST_VALUES['target'])
@pytest.mark.parametrize("start", TEST_VALUES['start'])
@pytest.mark.parametrize(
    "use_prediction_features", TEST_VALUES['use_prediction_features']
)
@pytest.mark.parametrize(
    "allow_target_padding", TEST_VALUES['allow_target_padding']
)
def test_CanonicalInstanceSplitter(
    start, target, is_train, use_prediction_features, allow_target_padding
):
    train_length = 100
    pred_length = 13
    t = transform.CanonicalInstanceSplitter(
        target_field=transform.FieldName.TARGET,
        is_pad_field=transform.FieldName.IS_PAD,
        start_field=transform.FieldName.START,
        forecast_start_field=transform.FieldName.FORECAST_START,
        instance_sampler=transform.UniformSplitSampler(p=1.0),
        instance_length=train_length,
        prediction_length=pred_length,
        time_series_fields=['some_time_feature'],
        allow_target_padding=allow_target_padding,
        use_prediction_features=use_prediction_features,
    )

    assert_serializable(t)

    other_feat = np.arange(len(target) + 100)
    data = {
        'start': start,
        'target': target,
        'some_time_feature': other_feat,
        'some_other_col': 'ABC',
    }

    out = list(t.flatmap_transform(data, is_train=is_train))

    min_num_instances = 1 if allow_target_padding else 0
    if is_train:
        assert len(out) == max(
            min_num_instances, len(target) - train_length + 1
        )
    else:
        assert len(out) == 1

    for o in out:
        assert 'target' not in o
        assert 'future_target' not in o
        assert 'some_time_feature' not in o
        assert 'some_other_col' in o

        assert len(o['past_some_time_feature']) == train_length
        assert len(o['past_target']) == train_length

        if use_prediction_features and not is_train:
            assert len(o['future_some_time_feature']) == pred_length


def test_Transformation():
    train_length = 100
    ds = gluonts.dataset.common.ListDataset(
        [{'start': '2012-01-01', 'target': [0.2] * train_length}], freq='1D'
    )

    pred_length = 10

    t = transform.Chain(
        trans=[
            transform.AddTimeFeatures(
                start_field=transform.FieldName.START,
                target_field=transform.FieldName.TARGET,
                output_field='time_feat',
                time_features=[
                    time_feature.DayOfWeek(),
                    time_feature.DayOfMonth(),
                    time_feature.MonthOfYear(),
                ],
                pred_length=pred_length,
            ),
            transform.AddAgeFeature(
                target_field=transform.FieldName.TARGET,
                output_field='age',
                pred_length=pred_length,
                log_scale=True,
            ),
            transform.AddObservedValuesIndicator(
                target_field=transform.FieldName.TARGET,
                output_field='observed_values',
            ),
            transform.VstackFeatures(
                output_field='dynamic_feat',
                input_fields=['age', 'time_feat'],
                drop_inputs=True,
            ),
            transform.InstanceSplitter(
                target_field=transform.FieldName.TARGET,
                is_pad_field=transform.FieldName.IS_PAD,
                start_field=transform.FieldName.START,
                forecast_start_field=transform.FieldName.FORECAST_START,
                train_sampler=transform.ExpectedNumInstanceSampler(
                    num_instances=4
                ),
                past_length=train_length,
                future_length=pred_length,
                time_series_fields=['dynamic_feat', 'observed_values'],
            ),
        ]
    )

    assert_serializable(t)

    for u in t(iter(ds), is_train=True):
        print(u)


@pytest.mark.parametrize("is_train", TEST_VALUES['is_train'])
def test_multi_dim_transformation(is_train):
    train_length = 10

    first_dim = np.arange(1, 11, 1).tolist()
    first_dim[-1] = "NaN"

    second_dim = np.arange(11, 21, 1).tolist()
    second_dim[0] = "NaN"

    ds = gluonts.dataset.common.ListDataset(
        data_iter=[{'start': '2012-01-01', 'target': [first_dim, second_dim]}],
        freq='1D',
        one_dim_target=False,
    )
    pred_length = 2

    # Looks weird - but this is necessary to assert the nan entries correctly.
    first_dim[-1] = np.nan
    second_dim[0] = np.nan

    t = transform.Chain(
        trans=[
            transform.AddTimeFeatures(
                start_field=transform.FieldName.START,
                target_field=transform.FieldName.TARGET,
                output_field='time_feat',
                time_features=[
                    time_feature.DayOfWeek(),
                    time_feature.DayOfMonth(),
                    time_feature.MonthOfYear(),
                ],
                pred_length=pred_length,
            ),
            transform.AddAgeFeature(
                target_field=transform.FieldName.TARGET,
                output_field='age',
                pred_length=pred_length,
                log_scale=True,
            ),
            transform.AddObservedValuesIndicator(
                target_field=transform.FieldName.TARGET,
                output_field='observed_values',
                convert_nans=False,
            ),
            transform.VstackFeatures(
                output_field='dynamic_feat',
                input_fields=['age', 'time_feat'],
                drop_inputs=True,
            ),
            transform.InstanceSplitter(
                target_field=transform.FieldName.TARGET,
                is_pad_field=transform.FieldName.IS_PAD,
                start_field=transform.FieldName.START,
                forecast_start_field=transform.FieldName.FORECAST_START,
                train_sampler=transform.ExpectedNumInstanceSampler(
                    num_instances=4
                ),
                past_length=train_length,
                future_length=pred_length,
                time_series_fields=['dynamic_feat', 'observed_values'],
                output_NTC=False,
            ),
        ]
    )

    assert_serializable(t)

    if is_train:
        for u in t(iter(ds), is_train=True):
            assert_shape(u['past_target'], (2, 10))
            assert_shape(u['past_dynamic_feat'], (4, 10))
            assert_shape(u['past_observed_values'], (2, 10))
            assert_shape(u['future_target'], (2, 2))

            assert_padded_array(
                u['past_observed_values'],
                np.array([[1.0] * 9 + [0.0], [0.0] + [1.0] * 9]),
                u['past_is_pad'],
            )
            assert_padded_array(
                u['past_target'],
                np.array([first_dim, second_dim]),
                u['past_is_pad'],
            )
    else:
        for u in t(iter(ds), is_train=False):
            assert_shape(u['past_target'], (2, 10))
            assert_shape(u['past_dynamic_feat'], (4, 10))
            assert_shape(u['past_observed_values'], (2, 10))
            assert_shape(u['future_target'], (2, 0))

            assert_padded_array(
                u['past_observed_values'],
                np.array([[1.0] * 9 + [0.0], [0.0] + [1.0] * 9]),
                u['past_is_pad'],
            )
            assert_padded_array(
                u['past_target'],
                np.array([first_dim, second_dim]),
                u['past_is_pad'],
            )


def test_ExpectedNumInstanceSampler():
    N = 6
    train_length = 2
    pred_length = 1
    ds = make_dataset(N, train_length)

    t = transform.Chain(
        trans=[
            transform.InstanceSplitter(
                target_field=transform.FieldName.TARGET,
                is_pad_field=transform.FieldName.IS_PAD,
                start_field=transform.FieldName.START,
                forecast_start_field=transform.FieldName.FORECAST_START,
                train_sampler=transform.ExpectedNumInstanceSampler(
                    num_instances=4
                ),
                past_length=train_length,
                future_length=pred_length,
                pick_incomplete=True,
            )
        ]
    )

    assert_serializable(t)

    scale_hist = ScaleHistogram()

    repetition = 2
    for i in range(repetition):
        for data in t(iter(ds), is_train=True):
            target_values = data['past_target']
            # for simplicity, discard values that are zeros to avoid confusion with padding
            target_values = target_values[target_values > 0]
            scale_hist.add(target_values)

    expected_values = {i: 2 ** i * repetition for i in range(1, N)}

    assert expected_values == scale_hist.bin_counts


def test_BucketInstanceSampler():
    N = 6
    train_length = 2
    pred_length = 1
    ds = make_dataset(N, train_length)

    dataset_stats = calculate_dataset_statistics(ds)

    t = transform.Chain(
        trans=[
            transform.InstanceSplitter(
                target_field=transform.FieldName.TARGET,
                is_pad_field=transform.FieldName.IS_PAD,
                start_field=transform.FieldName.START,
                forecast_start_field=transform.FieldName.FORECAST_START,
                train_sampler=transform.BucketInstanceSampler(
                    dataset_stats.scale_histogram
                ),
                past_length=train_length,
                future_length=pred_length,
                pick_incomplete=True,
            )
        ]
    )

    assert_serializable(t)

    scale_hist = ScaleHistogram()

    repetition = 200
    for i in range(repetition):
        for data in t(iter(ds), is_train=True):
            target_values = data['past_target']
            # for simplicity, discard values that are zeros to avoid confusion with padding
            target_values = target_values[target_values > 0]
            scale_hist.add(target_values)

    expected_values = {i: repetition for i in range(1, N)}
    found_values = scale_hist.bin_counts

    for i in range(1, N):
        assert abs(
            expected_values[i] - found_values[i] < expected_values[i] * 0.3
        )


def make_dataset(N, train_length):
    # generates 2 ** N - 1 timeseries with constant increasing values
    n = 2 ** N - 1
    targets = np.ones((n, train_length))
    for i in range(0, n):
        targets[i, :] = targets[i, :] * i

    ds = gluonts.dataset.common.ListDataset(
        data_iter=[
            {'start': '2012-01-01', 'target': targets[i, :]} for i in range(n)
        ],
        freq='1D',
    )

    return ds


def assert_serializable(x: transform.Transformation):
    t = fqname_for(x.__class__)
    y = load_json(dump_json(x))
    z = load_code(dump_code(x))
    assert dump_json(x) == dump_json(
        y
    ), f'Code serialization for transformer {t} does not work'
    assert dump_code(x) == dump_code(
        z
    ), f'JSON serialization for transformer {t} does not work'


def assert_shape(array: np.array, reference_shape: Tuple[int, int]):
    assert (
        array.shape == reference_shape
    ), f"Shape should be {reference_shape} but found {array.shape}."


def assert_padded_array(
    sampled_array: np.array, reference_array: np.array, padding_array: np.array
):
    num_padded = int(np.sum(padding_array))
    sampled_no_padding = sampled_array[:, num_padded:]

    reference_array = np.roll(reference_array, num_padded, axis=1)
    reference_no_padding = reference_array[:, num_padded:]

    # Convert nans to dummy value for assertion because
    # np.nan == np.nan -> False.
    reference_no_padding[np.isnan(reference_no_padding)] = 9999.0
    sampled_no_padding[np.isnan(sampled_no_padding)] = 9999.0

    reference_no_padding = np.array(reference_no_padding, dtype=np.float32)

    assert (sampled_no_padding == reference_no_padding).all(), (
        f"Sampled and reference arrays do not match. '"
        f"Got {sampled_no_padding} but should be {reference_no_padding}."
    )

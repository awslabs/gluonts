# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import Tuple

# Third-party imports
import numpy as np
import pandas as pd
import mxnet as mx
import pytest

# First-party imports
import gluonts
from gluonts import time_feature, transform
from gluonts.core import fqname_for
from gluonts.core.serde import dump_code, dump_json, load_code, load_json
from gluonts.dataset.common import ProcessStartField, DataEntry, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.stat import ScaleHistogram, calculate_dataset_statistics

FREQ = "1D"

TEST_VALUES = {
    "is_train": [True, False],
    "target": [np.zeros(0), np.random.rand(13), np.random.rand(100)],
    "start": [
        ProcessStartField.process("2012-01-02", freq="1D"),
        ProcessStartField.process("1994-02-19 20:01:02", freq="3D"),
    ],
    "use_prediction_features": [True, False],
    "allow_target_padding": [True, False],
    "lead_time": [0, 1, 10, 20],
}


def test_align_timestamp():
    def aligned_with(date_str, freq):
        return str(ProcessStartField.process(date_str, freq=freq))

    for _ in range(2):
        assert (
            aligned_with("2012-03-05 09:13:12", "min") == "2012-03-05 09:13:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "2min")
            == "2012-03-05 09:12:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "H") == "2012-03-05 09:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "D") == "2012-03-05 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "W") == "2012-03-11 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "4W") == "2012-03-11 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "M") == "2012-03-31 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "3M") == "2012-03-31 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:13:12", "Y") == "2012-12-31 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "min") == "2012-03-05 09:14:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "2min")
            == "2012-03-05 09:14:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "H") == "2012-03-05 09:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "D") == "2012-03-05 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "W") == "2012-03-11 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "4W") == "2012-03-11 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "M") == "2012-03-31 00:00:00"
        )
        assert (
            aligned_with("2012-03-05 09:14:11", "3M") == "2012-03-31 00:00:00"
        )


@pytest.mark.parametrize("is_train", TEST_VALUES["is_train"])
@pytest.mark.parametrize("target", TEST_VALUES["target"])
@pytest.mark.parametrize("start", TEST_VALUES["start"])
def test_AddTimeFeatures(start, target, is_train: bool):
    pred_length = 13
    t = transform.AddTimeFeatures(
        start_field=FieldName.START,
        target_field=FieldName.TARGET,
        output_field="myout",
        pred_length=pred_length,
        time_features=[time_feature.DayOfWeek(), time_feature.DayOfMonth()],
    )

    assert_serializable(t)

    data = {"start": start, "target": target}
    res = t.map_transform(data, is_train=is_train)
    mat = res["myout"]
    expected_length = len(target) + (0 if is_train else pred_length)
    assert mat.shape == (2, expected_length)
    tmp_idx = pd.date_range(
        start=start, freq=start.freq, periods=expected_length
    )
    assert np.alltrue(mat[0] == time_feature.DayOfWeek()(tmp_idx))
    assert np.alltrue(mat[1] == time_feature.DayOfMonth()(tmp_idx))


@pytest.mark.parametrize("is_train", TEST_VALUES["is_train"])
@pytest.mark.parametrize("target", TEST_VALUES["target"])
@pytest.mark.parametrize("start", TEST_VALUES["start"])
def test_AddTimeFeatures_empty_time_features(start, target, is_train: bool):
    pred_length = 13
    t = transform.AddTimeFeatures(
        start_field=FieldName.START,
        target_field=FieldName.TARGET,
        output_field="myout",
        pred_length=pred_length,
        time_features=[],
    )

    assert_serializable(t)

    data = {"start": start, "target": target}
    res = t.map_transform(data, is_train=is_train)
    assert res["myout"] is None


@pytest.mark.parametrize("is_train", TEST_VALUES["is_train"])
@pytest.mark.parametrize("target", TEST_VALUES["target"])
@pytest.mark.parametrize("start", TEST_VALUES["start"])
def test_AddAgeFeatures(start, target, is_train: bool):
    pred_length = 13
    t = transform.AddAgeFeature(
        pred_length=pred_length,
        target_field=FieldName.TARGET,
        output_field="age",
        log_scale=True,
    )

    assert_serializable(t)

    data = {"start": start, "target": target}
    out = t.map_transform(data, is_train=is_train)
    expected_length = len(target) + (0 if is_train else pred_length)
    assert out["age"].shape[-1] == expected_length
    assert np.allclose(
        out["age"],
        np.log10(2.0 + np.arange(expected_length)).reshape(
            (1, expected_length)
        ),
    )


@pytest.mark.parametrize(
    "pick_incomplete", TEST_VALUES["allow_target_padding"]
)
@pytest.mark.parametrize("is_train", TEST_VALUES["is_train"])
@pytest.mark.parametrize("target", TEST_VALUES["target"])
@pytest.mark.parametrize("start", TEST_VALUES["start"])
@pytest.mark.parametrize("lead_time", TEST_VALUES["lead_time"])
def test_InstanceSplitter(
    start, target, lead_time: int, is_train: bool, pick_incomplete: bool
):
    train_length = 100
    pred_length = 13
    t = transform.InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        train_sampler=transform.UniformSplitSampler(p=1.0),
        past_length=train_length,
        future_length=pred_length,
        lead_time=lead_time,
        time_series_fields=["some_time_feature"],
        pick_incomplete=pick_incomplete,
    )

    assert_serializable(t)

    other_feat = np.arange(len(target) + 100)
    data = {
        "start": start,
        "target": target,
        "some_time_feature": other_feat,
        "some_other_col": "ABC",
    }

    if not is_train and not pick_incomplete and len(target) < train_length:
        with pytest.raises(AssertionError):
            out = list(t.flatmap_transform(data, is_train=is_train))
        return
    else:
        out = list(t.flatmap_transform(data, is_train=is_train))

    if is_train:
        assert len(out) == max(
            0,
            len(target)
            - pred_length
            - lead_time
            + 1
            - (0 if pick_incomplete else train_length),
        )
    else:
        assert len(out) == 1

    for o in out:
        assert "target" not in o
        assert "some_time_feature" not in o
        assert "some_other_col" in o

        assert len(o["past_some_time_feature"]) == train_length
        assert len(o["past_target"]) == train_length

        if is_train:
            assert len(o["future_target"]) == pred_length
            assert len(o["future_some_time_feature"]) == pred_length
        else:
            assert len(o["future_target"]) == 0
            assert len(o["future_some_time_feature"]) == pred_length

    # expected_length = len(target) + (0 if is_train else pred_length)
    # assert len(out['age']) == expected_length
    # assert np.alltrue(out['age'] == np.log10(2.0 + np.arange(expected_length)))


@pytest.mark.parametrize("is_train", TEST_VALUES["is_train"])
@pytest.mark.parametrize("target", TEST_VALUES["target"])
@pytest.mark.parametrize("start", TEST_VALUES["start"])
@pytest.mark.parametrize(
    "use_prediction_features", TEST_VALUES["use_prediction_features"]
)
@pytest.mark.parametrize(
    "allow_target_padding", TEST_VALUES["allow_target_padding"]
)
def test_CanonicalInstanceSplitter(
    start,
    target,
    is_train: bool,
    use_prediction_features: bool,
    allow_target_padding: bool,
):
    train_length = 100
    pred_length = 13
    t = transform.CanonicalInstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=transform.UniformSplitSampler(p=1.0),
        instance_length=train_length,
        prediction_length=pred_length,
        time_series_fields=["some_time_feature"],
        allow_target_padding=allow_target_padding,
        use_prediction_features=use_prediction_features,
    )

    assert_serializable(t)

    other_feat = np.arange(len(target) + 100)
    data = {
        "start": start,
        "target": target,
        "some_time_feature": other_feat,
        "some_other_col": "ABC",
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
        assert "target" not in o
        assert "future_target" not in o
        assert "some_time_feature" not in o
        assert "some_other_col" in o

        assert len(o["past_some_time_feature"]) == train_length
        assert len(o["past_target"]) == train_length

        if use_prediction_features and not is_train:
            assert len(o["future_some_time_feature"]) == pred_length


def test_Transformation():
    train_length = 100
    ds = gluonts.dataset.common.ListDataset(
        [{"start": "2012-01-01", "target": [0.2] * train_length}], freq="1D"
    )

    pred_length = 10

    t = transform.Chain(
        trans=[
            transform.AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field="time_feat",
                time_features=[
                    time_feature.DayOfWeek(),
                    time_feature.DayOfMonth(),
                    time_feature.MonthOfYear(),
                ],
                pred_length=pred_length,
            ),
            transform.AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field="age",
                pred_length=pred_length,
                log_scale=True,
            ),
            transform.AddObservedValuesIndicator(
                target_field=FieldName.TARGET, output_field="observed_values"
            ),
            transform.VstackFeatures(
                output_field="dynamic_feat",
                input_fields=["age", "time_feat"],
                drop_inputs=True,
            ),
            transform.InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=transform.ExpectedNumInstanceSampler(
                    num_instances=4
                ),
                past_length=train_length,
                future_length=pred_length,
                time_series_fields=["dynamic_feat", "observed_values"],
            ),
        ]
    )

    assert_serializable(t)

    for u in t(iter(ds), is_train=True):
        print(u)


@pytest.mark.parametrize("is_train", TEST_VALUES["is_train"])
def test_multi_dim_transformation(is_train):
    train_length = 10

    first_dim: list = list(np.arange(1, 11, 1))
    first_dim[-1] = "NaN"

    second_dim: list = list(np.arange(11, 21, 1))
    second_dim[0] = "NaN"

    ds = gluonts.dataset.common.ListDataset(
        data_iter=[{"start": "2012-01-01", "target": [first_dim, second_dim]}],
        freq="1D",
        one_dim_target=False,
    )
    pred_length = 2

    # Looks weird - but this is necessary to assert the nan entries correctly.
    first_dim[-1] = np.nan
    second_dim[0] = np.nan

    t = transform.Chain(
        trans=[
            transform.AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field="time_feat",
                time_features=[
                    time_feature.DayOfWeek(),
                    time_feature.DayOfMonth(),
                    time_feature.MonthOfYear(),
                ],
                pred_length=pred_length,
            ),
            transform.AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field="age",
                pred_length=pred_length,
                log_scale=True,
            ),
            transform.AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field="observed_values",
                convert_nans=False,
            ),
            transform.VstackFeatures(
                output_field="dynamic_feat",
                input_fields=["age", "time_feat"],
                drop_inputs=True,
            ),
            transform.InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=transform.ExpectedNumInstanceSampler(
                    num_instances=4
                ),
                past_length=train_length,
                future_length=pred_length,
                time_series_fields=["dynamic_feat", "observed_values"],
                output_NTC=False,
            ),
        ]
    )

    assert_serializable(t)

    if is_train:
        for u in t(iter(ds), is_train=True):
            assert_shape(u["past_target"], (2, 10))
            assert_shape(u["past_dynamic_feat"], (4, 10))
            assert_shape(u["past_observed_values"], (2, 10))
            assert_shape(u["future_target"], (2, 2))

            assert_padded_array(
                u["past_observed_values"],
                np.array([[1.0] * 9 + [0.0], [0.0] + [1.0] * 9]),
                u["past_is_pad"],
            )
            assert_padded_array(
                u["past_target"],
                np.array([first_dim, second_dim]),
                u["past_is_pad"],
            )
    else:
        for u in t(iter(ds), is_train=False):
            assert_shape(u["past_target"], (2, 10))
            assert_shape(u["past_dynamic_feat"], (4, 10))
            assert_shape(u["past_observed_values"], (2, 10))
            assert_shape(u["future_target"], (2, 0))

            assert_padded_array(
                u["past_observed_values"],
                np.array([[1.0] * 9 + [0.0], [0.0] + [1.0] * 9]),
                u["past_is_pad"],
            )
            assert_padded_array(
                u["past_target"],
                np.array([first_dim, second_dim]),
                u["past_is_pad"],
            )


def test_ExpectedNumInstanceSampler():
    N = 6
    train_length = 2
    pred_length = 1
    ds = make_dataset(N, train_length)

    t = transform.Chain(
        trans=[
            transform.InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
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
            target_values = data["past_target"]
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
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
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
            target_values = data["past_target"]
            # for simplicity, discard values that are zeros to avoid confusion with padding
            target_values = target_values[target_values > 0]
            scale_hist.add(target_values)

    expected_values = {i: repetition for i in range(1, N)}
    found_values = scale_hist.bin_counts

    for i in range(1, N):
        assert abs(
            expected_values[i] - found_values[i] < expected_values[i] * 0.3
        )


def test_cdf_to_gaussian_transformation():
    def make_test_data():
        target = np.array(
            [
                0,
                0,
                0,
                0,
                10,
                10,
                20,
                20,
                30,
                30,
                40,
                50,
                59,
                60,
                60,
                70,
                80,
                90,
                100,
            ]
        ).tolist()

        np.random.shuffle(target)

        multi_dim_target = np.array([target, target]).transpose()

        past_is_pad = np.array([[0] * len(target)]).transpose()

        past_observed_target = np.array(
            [[1] * len(target), [1] * len(target)]
        ).transpose()

        ds = gluonts.dataset.common.ListDataset(
            # Mimic output from InstanceSplitter
            data_iter=[
                {
                    "start": "2012-01-01",
                    "target": multi_dim_target,
                    "past_target": multi_dim_target,
                    "future_target": multi_dim_target,
                    "past_is_pad": past_is_pad,
                    f"past_{FieldName.OBSERVED_VALUES}": past_observed_target,
                }
            ],
            freq="1D",
            one_dim_target=False,
        )
        return ds

    def make_fake_output(u: DataEntry):
        fake_output = np.expand_dims(
            np.expand_dims(u["past_target_cdf"], axis=0), axis=0
        )
        return fake_output

    ds = make_test_data()

    t = transform.Chain(
        trans=[
            transform.CDFtoGaussianTransform(
                target_field=FieldName.TARGET,
                observed_values_field=FieldName.OBSERVED_VALUES,
                max_context_length=20,
                target_dim=2,
            )
        ]
    )

    for u in t(iter(ds), is_train=False):

        fake_output = make_fake_output(u)

        # Fake transformation chain output
        u["past_target_sorted"] = mx.nd.array(
            np.expand_dims(u["past_target_sorted"], axis=0)
        )

        u["slopes"] = mx.nd.array(np.expand_dims(u["slopes"], axis=0))

        u["intercepts"] = mx.nd.array(np.expand_dims(u["intercepts"], axis=0))

        back_transformed = transform.cdf_to_gaussian_forward_transform(
            u, fake_output
        )

        # Get any sample/batch (slopes[i][:, d]they are all the same)
        back_transformed = back_transformed[0][0]

        original_target = u["target"]

        # Original target and back-transformed target should be the same
        assert np.allclose(original_target, back_transformed)


def test_gaussian_cdf():
    try:
        from scipy.stats import norm
    except:
        pytest.skip("scipy not installed skipping test for erf")

    x = np.array(
        [-1000, -100, -10]
        + np.linspace(-2, 2, 1001).tolist()
        + [10, 100, 1000]
    )
    y_gluonts = transform.CDFtoGaussianTransform.standard_gaussian_cdf(x)
    y_scipy = norm.cdf(x)

    assert np.allclose(y_gluonts, y_scipy, atol=1e-7)


def test_gaussian_ppf():
    try:
        from scipy.stats import norm
    except:
        pytest.skip("scipy not installed skipping test for erf")

    x = np.linspace(0.0001, 0.9999, 1001)
    y_gluonts = transform.CDFtoGaussianTransform.standard_gaussian_ppf(x)
    y_scipy = norm.ppf(x)

    assert np.allclose(y_gluonts, y_scipy, atol=1e-7)


def test_target_dim_indicator():
    target = np.array([0, 2, 3, 10]).tolist()

    multi_dim_target = np.array([target, target, target, target])
    dataset = gluonts.dataset.common.ListDataset(
        data_iter=[{"start": "2012-01-01", "target": multi_dim_target}],
        freq="1D",
        one_dim_target=False,
    )

    t = transform.Chain(
        trans=[
            transform.TargetDimIndicator(
                target_field=FieldName.TARGET, field_name="target_dimensions"
            )
        ]
    )

    for data_entry in t(dataset, is_train=True):
        assert (
            data_entry["target_dimensions"] == np.array([0, 1, 2, 3])
        ).all()


@pytest.fixture
def point_process_dataset():

    ia_times = np.array([0.2, 0.7, 0.2, 0.5, 0.3, 0.3, 0.2, 0.1])
    marks = np.array([0, 1, 2, 0, 1, 2, 2, 2])

    lds = ListDataset(
        [
            {
                "target": np.c_[ia_times, marks].T,
                "start": pd.Timestamp("2011-01-01 00:00:00", freq="H"),
                "end": pd.Timestamp("2011-01-01 03:00:00", freq="H"),
            }
        ],
        freq="H",
        one_dim_target=False,
    )

    return lds


class MockContinuousTimeSampler(transform.ContinuousTimePointSampler):
    # noinspection PyMissingConstructor,PyUnusedLocal
    def __init__(self, ret_values, *args, **kwargs):
        self._ret_values = ret_values

    def __call__(self, *args, **kwargs):
        return np.array(self._ret_values)


def test_ctsplitter_mask_sorted(point_process_dataset):
    d = next(iter(point_process_dataset))

    ia_times = d["target"][0, :]

    ts = np.cumsum(ia_times)

    splitter = transform.ContinuousTimeInstanceSplitter(
        2,
        1,
        train_sampler=transform.ContinuousTimeUniformSampler(num_instances=10),
    )

    # no boundary conditions
    res = splitter._mask_sorted(ts, 1, 2)
    assert all([a == b for a, b in zip([2, 3, 4], res)])

    # lower bound equal, exclusive of upper bound
    res = splitter._mask_sorted(np.array([1, 2, 3, 4, 5, 6]), 1, 2)
    assert all([a == b for a, b in zip([0], res)])


def test_ctsplitter_no_train_last_point(point_process_dataset):
    splitter = transform.ContinuousTimeInstanceSplitter(
        2,
        1,
        train_sampler=transform.ContinuousTimeUniformSampler(num_instances=10),
    )

    iter_de = splitter(point_process_dataset, is_train=False)

    d_out = next(iter(iter_de))

    assert "future_target" not in d_out
    assert "future_valid_length" not in d_out
    assert "past_target" in d_out
    assert "past_valid_length" in d_out

    assert d_out["past_valid_length"] == 6
    assert np.allclose(
        [0.1, 0.5, 0.3, 0.3, 0.2, 0.1], d_out["past_target"][..., 0], atol=0.01
    )


def test_ctsplitter_train_correct(point_process_dataset):
    splitter = transform.ContinuousTimeInstanceSplitter(
        1,
        1,
        train_sampler=MockContinuousTimeSampler(
            ret_values=[1.01, 1.5, 1.99], num_instances=3
        ),
    )

    iter_de = splitter(point_process_dataset, is_train=True)

    outputs = list(iter_de)

    assert outputs[0]["past_valid_length"] == 2
    assert outputs[0]["future_valid_length"] == 3

    assert np.allclose(
        outputs[0]["past_target"], np.array([[0.19, 0.7], [0, 1]]).T
    )
    assert np.allclose(
        outputs[0]["future_target"], np.array([[0.09, 0.5, 0.3], [2, 0, 1]]).T
    )

    assert outputs[1]["past_valid_length"] == 2
    assert outputs[1]["future_valid_length"] == 4

    assert outputs[2]["past_valid_length"] == 3
    assert outputs[2]["future_valid_length"] == 3


def test_ctsplitter_train_correct_out_count(point_process_dataset):

    # produce new TPP data by shuffling existing TS instance
    def shuffle_iterator(num_duplications=5):
        for entry in point_process_dataset:
            for i in range(num_duplications):
                d = dict.copy(entry)
                d["target"] = np.random.permutation(d["target"].T).T
                yield d

    splitter = transform.ContinuousTimeInstanceSplitter(
        1,
        1,
        train_sampler=MockContinuousTimeSampler(
            ret_values=[1.01, 1.5, 1.99], num_instances=3
        ),
    )

    iter_de = splitter(shuffle_iterator(), is_train=True)

    outputs = list(iter_de)

    assert len(outputs) == 5 * 3


def test_ctsplitter_train_samples_correct_times(point_process_dataset):

    splitter = transform.ContinuousTimeInstanceSplitter(
        1.25, 1.25, train_sampler=transform.ContinuousTimeUniformSampler(20)
    )

    iter_de = splitter(point_process_dataset, is_train=True)

    assert all(
        [
            (
                pd.Timestamp("2011-01-01 01:15:00")
                <= d["forecast_start"]
                <= pd.Timestamp("2011-01-01 01:45:00")
            )
            for d in iter_de
        ]
    )


def test_ctsplitter_train_short_intervals(point_process_dataset):
    splitter = transform.ContinuousTimeInstanceSplitter(
        0.01,
        0.01,
        train_sampler=MockContinuousTimeSampler(
            ret_values=[1.01, 1.5, 1.99], num_instances=3
        ),
    )

    iter_de = splitter(point_process_dataset, is_train=True)

    for d in iter_de:
        assert d["future_valid_length"] == d["past_valid_length"] == 0
        assert np.prod(np.shape(d["past_target"])) == 0
        assert np.prod(np.shape(d["future_target"])) == 0


def make_dataset(N, train_length):
    # generates 2 ** N - 1 timeseries with constant increasing values
    n = 2 ** N - 1
    targets = np.ones((n, train_length))
    for i in range(0, n):
        targets[i, :] = targets[i, :] * i

    ds = gluonts.dataset.common.ListDataset(
        data_iter=[
            {"start": "2012-01-01", "target": targets[i, :]} for i in range(n)
        ],
        freq="1D",
    )

    return ds


def assert_serializable(x: transform.Transformation):
    t = fqname_for(x.__class__)
    y = load_json(dump_json(x))
    z = load_code(dump_code(x))
    assert dump_json(x) == dump_json(
        y
    ), f"Code serialization for transformer {t} does not work"
    assert dump_code(x) == dump_code(
        z
    ), f"JSON serialization for transformer {t} does not work"


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

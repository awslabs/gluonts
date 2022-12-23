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

from gluonts.core.serde.flat import clone
from gluonts import time_feature, transform
from gluonts.core.component import equals
from gluonts.dataset.field_names import FieldName
from gluonts.mx import CanonicalRNNEstimator

from pandas.tseries.frequencies import to_offset


def test_map_transformation():
    tran = transform.VstackFeatures(
        output_field="dynamic_feat",
        input_fields=["age", "time_feat"],
        drop_inputs=True,
    )

    assert equals(tran, clone(tran))
    assert not equals(tran, clone(tran, {"drop_inputs": False}))


def test_add_time_features():
    tran = transform.AddTimeFeatures(
        start_field=FieldName.START,
        target_field=FieldName.TARGET,
        output_field="time_feat",
        time_features=[
            time_feature.day_of_week,
            time_feature.day_of_month,
            time_feature.month_of_year,
        ],
        pred_length=10,
    )

    tran2 = clone(
        tran,
        {
            "time_features": [
                time_feature.day_of_week,
                time_feature.day_of_month,
            ]
        },
    )

    assert equals(tran, clone(tran))
    assert not equals(tran, tran2)


def test_filter_transformation():
    prediction_length = 10
    tran1 = transform.FilterTransformation(
        lambda x: x["target"].shape[-1] > prediction_length
    )
    # serde.flat.clone(tran1) does not work on
    tran2 = transform.FilterTransformation(
        lambda x: x["target"].shape[-1] > prediction_length
    )
    tran3 = transform.FilterTransformation(
        condition=lambda x: x["target"].shape[-1] < prediction_length
    )

    assert equals(tran1, tran2)
    assert not equals(tran1, tran3)


def test_exp_num_sampler():
    sampler = transform.ExpectedNumInstanceSampler(num_instances=4)
    assert equals(sampler, clone(sampler))
    assert not equals(sampler, clone(sampler, {"num_instances": 5}))


def test_continuous_time_sampler():
    sampler = transform.ContinuousTimeUniformSampler(num_instances=4)
    assert equals(sampler, clone(sampler))
    assert not equals(sampler, clone(sampler, {"num_instances": 5}))


def test_instance_splitter():
    splitter = transform.InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=transform.ExpectedNumInstanceSampler(num_instances=4),
        past_length=100,
        future_length=10,
        time_series_fields=["dynamic_feat", "observed_values"],
    )

    splitter2 = clone(
        splitter,
        {
            "instance_sampler": transform.ExpectedNumInstanceSampler(
                num_instances=5
            )
        },
    )
    assert equals(splitter, clone(splitter))
    assert not equals(splitter, splitter2)


def test_continuous_time_splitter():
    splitter = transform.ContinuousTimeInstanceSplitter(
        past_interval_length=1,
        future_interval_length=1,
        instance_sampler=transform.ContinuousTimePointSampler(),
        freq=to_offset("H"),
    )

    splitter2 = transform.ContinuousTimeInstanceSplitter(
        past_interval_length=1,
        future_interval_length=1,
        instance_sampler=transform.ContinuousTimePointSampler(min_past=1.0),
        freq=to_offset("H"),
    )

    assert equals(splitter, clone(splitter))
    assert not equals(splitter, splitter2)


def test_chain():
    chain = transform.Chain(
        trans=[
            transform.AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field="time_feat",
                time_features=[
                    time_feature.day_of_week,
                    time_feature.day_of_month,
                    time_feature.month_of_year,
                ],
                pred_length=10,
            ),
            transform.AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field="age",
                pred_length=10,
                log_scale=True,
            ),
            transform.AddObservedValuesIndicator(
                target_field=FieldName.TARGET, output_field="observed_values"
            ),
        ]
    )

    assert equals(chain, clone(chain))
    assert not equals(chain, clone(chain, {"trans": []}))

    another_chain = transform.Chain(
        trans=[
            transform.AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field="time_feat",
                time_features=[
                    time_feature.day_of_week,
                    time_feature.day_of_month,
                    time_feature.month_of_year,
                ],
                pred_length=10,
            ),
            transform.AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field="age",
                pred_length=10,
                log_scale=False,
            ),
            transform.AddObservedValuesIndicator(
                target_field=FieldName.TARGET, output_field="observed_values"
            ),
        ]
    )
    assert not equals(chain, another_chain)


def test_gluon_predictor():
    train_length = 100
    pred_length = 10

    estimator = CanonicalRNNEstimator("5min", train_length, pred_length)

    assert equals(estimator, clone(estimator))
    assert not equals(estimator, clone(estimator, {"freq": "1h"}))

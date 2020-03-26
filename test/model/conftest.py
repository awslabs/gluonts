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

import tempfile
from pathlib import Path

import pytest

try:
    import statsmodels
except ImportError:
    statsmodels = None


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pytest_runtest_setup(item):
    skip_datasets = [
        mark.args[0] for mark in item.iter_markers(name="skip_dataset")
    ]

    if skip_datasets:
        ds_name = item._request.getfixturevalue("dsinfo")["name"]
        if ds_name in skip_datasets:
            pytest.skip(f"Skip test on dataset {ds_name}")


@pytest.fixture(scope="session", params=["synthetic", "constant"])
def dsinfo(request):
    from gluonts import time_feature
    from gluonts.dataset.artificial import constant_dataset, default_synthetic

    if request.param == "constant":
        ds_info, train_ds, test_ds = constant_dataset()

        return AttrDict(
            name="constant",
            cardinality=int(ds_info.metadata.feat_static_cat[0].cardinality),
            freq=ds_info.metadata.freq,
            num_parallel_samples=2,
            prediction_length=ds_info.prediction_length,
            # FIXME: Should time features should not be needed for GP
            time_features=[time_feature.DayOfWeek(), time_feature.HourOfDay()],
            train_ds=train_ds,
            test_ds=test_ds,
        )
    elif request.param == "synthetic":
        ds_info, train_ds, test_ds = default_synthetic()

        return AttrDict(
            name="synthetic",
            batch_size=32,
            cardinality=int(ds_info.metadata.feat_static_cat[0].cardinality),
            context_length=2,
            freq=ds_info.metadata.freq,
            prediction_length=ds_info.prediction_length,
            num_parallel_samples=2,
            train_ds=train_ds,
            test_ds=test_ds,
            time_features=None,
        )


def from_hyperparameters(Estimator, hyperparameters, dsinfo):
    return Estimator.from_hyperparameters(
        freq=dsinfo.freq,
        **{
            "prediction_length": dsinfo.prediction_length,
            "num_parallel_samples": dsinfo.num_parallel_samples,
        },
        **hyperparameters,
    )


@pytest.fixture()
def accuracy_test(dsinfo):
    from gluonts.evaluation import Evaluator
    from gluonts.evaluation.backtest import backtest_metrics

    def test_accuracy(Estimator, hyperparameters, accuracy):
        estimator = from_hyperparameters(Estimator, hyperparameters, dsinfo)
        agg_metrics, item_metrics = backtest_metrics(
            train_dataset=dsinfo.train_ds,
            test_dataset=dsinfo.test_ds,
            forecaster=estimator,
            evaluator=Evaluator(calculate_owa=statsmodels is not None),
        )

        if dsinfo.name == "synthetic":
            accuracy = 10.0

        assert agg_metrics["ND"] <= accuracy

    return test_accuracy


@pytest.fixture()
def serialize_test(dsinfo):
    from gluonts.model.predictor import Predictor

    def test_serialize(Estimator, hyperparameters):
        estimator = from_hyperparameters(Estimator, hyperparameters, dsinfo)

        with tempfile.TemporaryDirectory() as temp_dir:
            predictor_act = estimator.train(dsinfo.train_ds)
            predictor_act.serialize(Path(temp_dir))
            predictor_exp = Predictor.deserialize(Path(temp_dir))
            # TODO: DeepFactorEstimator does not pass this assert
            assert predictor_act == predictor_exp

    return test_serialize


@pytest.fixture()
def repr_test(dsinfo):
    from gluonts.core.serde import load_code

    def test_repr(Estimator, hyperparameters):
        estimator = from_hyperparameters(Estimator, hyperparameters, dsinfo)
        assert repr(estimator) == repr(load_code(repr(estimator)))

    return test_repr

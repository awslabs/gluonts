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

import pytest

from gluonts.model.seq2seq import (
    MQCNNEstimator,
    MQRNNEstimator,
)
from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        context_length=dsinfo.prediction_length,
        num_batches_per_epoch=1,
        quantiles=[0.1, 0.5, 0.9],
        use_symbol_block_predictor=True,
    )


@pytest.fixture(
    params=[MQCNNEstimator, MQRNNEstimator], ids=["mqcnn", "mqrnn"]
)
def Estimator(request):
    return request.param


@pytest.mark.parametrize("quantiles", [[0.1, 0.5, 0.9], [0.5]])
@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(
    Estimator, accuracy_test, hyperparameters, hybridize, quantiles
):
    hyperparameters.update(
        num_batches_per_epoch=100, hybridize=hybridize, quantiles=quantiles
    )

    accuracy_test(Estimator, hyperparameters, accuracy=0.30)


@pytest.mark.parametrize("use_feat_dynamic_real", [True, False])
@pytest.mark.parametrize("add_time_feature", [True, False])
@pytest.mark.parametrize("add_age_feature", [True, False])
@pytest.mark.parametrize("enable_decoder_dynamic_feature", [True, False])
@pytest.mark.parametrize("hybridize", [True, False])
def test_mqcnn_covariate_smoke_test(
    use_feat_dynamic_real,
    add_time_feature,
    add_age_feature,
    enable_decoder_dynamic_feature,
    hybridize,
):
    hps = {
        "seed": 42,
        "freq": "D",
        "prediction_length": 3,
        "quantiles": [0.5, 0.1],
        "epochs": 3,
        "num_batches_per_epoch": 3,
        "use_feat_dynamic_real": use_feat_dynamic_real,
        "add_time_feature": add_time_feature,
        "add_age_feature": add_age_feature,
        "enable_decoder_dynamic_feature": enable_decoder_dynamic_feature,
        "hybridize": hybridize,
    }

    dataset_train, dataset_test = make_dummy_datasets_with_features(
        cardinality=[3, 10],
        num_feat_dynamic_real=2,
        freq=hps["freq"],
        prediction_length=hps["prediction_length"],
    )

    estimator = MQCNNEstimator.from_hyperparameters(**hps)

    predictor = estimator.train(dataset_train, num_workers=0)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)


# Test scaling and from inputs
@pytest.mark.parametrize("scaling", [True, False])
def test_mqcnn_scaling_smoke_test(scaling):
    hps = {
        "seed": 42,
        "freq": "D",
        "prediction_length": 3,
        "quantiles": [0.5, 0.1],
        "epochs": 3,
        "num_batches_per_epoch": 3,
        "scaling": scaling,
    }

    dataset_train, dataset_test = make_dummy_datasets_with_features(
        cardinality=[3, 10],
        num_feat_dynamic_real=2,
        freq=hps["freq"],
        prediction_length=hps["prediction_length"],
    )

    estimator = MQCNNEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train, num_workers=0)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)


def test_repr(Estimator, repr_test, hyperparameters):
    repr_test(Estimator, hyperparameters)


def test_serialize(Estimator, serialize_test, hyperparameters):
    serialize_test(Estimator, hyperparameters)

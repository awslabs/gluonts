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

from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator
from gluonts.mx.distribution import GaussianOutput
from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        context_length=dsinfo.prediction_length,
        num_forking=1,
        num_batches_per_epoch=1,
        use_symbol_block_predictor=True,
    )


@pytest.fixture(
    params=[MQCNNEstimator, MQRNNEstimator], ids=["mqcnn", "mqrnn"]
)
def Estimator(request):
    return request.param


@pytest.mark.parametrize("hybridize", [True, False])
@pytest.mark.parametrize(
    "quantiles, distr_output",
    [([0.1, 0.5, 0.9], None), (None, GaussianOutput())],
)
def test_accuracy(
    Estimator,
    accuracy_test,
    hyperparameters,
    hybridize,
    quantiles,
    distr_output,
):
    hyperparameters.update(
        num_batches_per_epoch=100,
        hybridize=hybridize,
        quantiles=quantiles,
        distr_output=distr_output,
    )

    accuracy_test(
        Estimator, hyperparameters, accuracy=0.20 if quantiles else 0.70
    )


@pytest.mark.parametrize("use_past_feat_dynamic_real", [True, False])
@pytest.mark.parametrize("use_feat_dynamic_real", [True, False])
@pytest.mark.parametrize("add_time_feature", [True, False])
@pytest.mark.parametrize("add_age_feature", [True, False])
@pytest.mark.parametrize("enable_encoder_dynamic_feature", [True, False])
@pytest.mark.parametrize("enable_decoder_dynamic_feature", [True, False])
@pytest.mark.parametrize("hybridize", [True, False])
@pytest.mark.parametrize(
    "quantiles, distr_output",
    [
        ([0.5, 0.1], None),
        (None, GaussianOutput()),
    ],
)
def test_mqcnn_covariate_smoke_test(
    use_past_feat_dynamic_real,
    use_feat_dynamic_real,
    add_time_feature,
    add_age_feature,
    enable_encoder_dynamic_feature,
    enable_decoder_dynamic_feature,
    hybridize,
    quantiles,
    distr_output,
):
    hps = {
        "seed": 42,
        "freq": "Y",
        "context_length": 5,
        "prediction_length": 3,
        "quantiles": quantiles,
        "distr_output": distr_output,
        "epochs": 3,
        "num_batches_per_epoch": 3,
        "use_past_feat_dynamic_real": use_past_feat_dynamic_real,
        "use_feat_dynamic_real": use_feat_dynamic_real,
        "add_time_feature": add_time_feature,
        "add_age_feature": add_age_feature,
        "enable_encoder_dynamic_feature": enable_encoder_dynamic_feature,
        "enable_decoder_dynamic_feature": enable_decoder_dynamic_feature,
        "hybridize": hybridize,
    }

    dataset_train, dataset_test = make_dummy_datasets_with_features(
        cardinality=[3, 10],
        num_feat_dynamic_real=2,
        num_past_feat_dynamic_real=4,
        freq=hps["freq"],
        prediction_length=hps["prediction_length"],
    )

    estimator = MQCNNEstimator.from_hyperparameters(**hps)

    predictor = estimator.train(dataset_train, num_workers=None)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)


@pytest.mark.parametrize("use_feat_static_cat", [True, False])
@pytest.mark.parametrize("cardinality", [[], [3, 10]])
def test_feat_static_cat_smoke_test(use_feat_static_cat, cardinality):
    hps = {
        "seed": 42,
        "freq": "D",
        "prediction_length": 3,
        "quantiles": [0.5, 0.1],
        "epochs": 3,
        "num_batches_per_epoch": 3,
        "use_feat_static_cat": use_feat_static_cat,
    }

    dataset_train, dataset_test = make_dummy_datasets_with_features(
        cardinality=cardinality,
        num_feat_dynamic_real=2,
        freq=hps["freq"],
        prediction_length=hps["prediction_length"],
    )
    estimator = MQCNNEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train, num_workers=None)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)


# Test scaling and from inputs
@pytest.mark.parametrize("scaling", [True, False])
@pytest.mark.parametrize("scaling_decoder_dynamic_feature", [True, False])
def test_mqcnn_scaling_smoke_test(scaling, scaling_decoder_dynamic_feature):
    hps = {
        "seed": 42,
        "freq": "D",
        "prediction_length": 3,
        "quantiles": [0.5, 0.1],
        "epochs": 3,
        "num_batches_per_epoch": 3,
        "scaling": scaling,
        "scaling_decoder_dynamic_feature": scaling_decoder_dynamic_feature,
    }

    dataset_train, dataset_test = make_dummy_datasets_with_features(
        cardinality=[3, 10],
        num_feat_dynamic_real=2,
        freq=hps["freq"],
        prediction_length=hps["prediction_length"],
    )

    estimator = MQCNNEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train, num_workers=None)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)


def test_repr(Estimator, repr_test, hyperparameters):
    repr_test(Estimator, hyperparameters)


def test_serialize(Estimator, serialize_test, hyperparameters):
    serialize_test(Estimator, hyperparameters)


def test_backwards_compatibility():
    hps = {
        "freq": "D",
        "context_length": 5,
        "num_forking": 4,
        "prediction_length": 3,
        "quantiles": [0.5, 0.1],
        "epochs": 3,
        "num_batches_per_epoch": 3,
        "use_feat_dynamic_real": True,
        "use_past_feat_dynamic_real": True,
        "enable_encoder_dynamic_feature": True,
        "enable_decoder_dynamic_feature": True,
        "scaling": True,
        "scaling_decoder_dynamic_feature": True,
    }

    dataset_train, dataset_test = make_dummy_datasets_with_features(
        cardinality=[3, 10],
        num_feat_dynamic_real=2,
        num_past_feat_dynamic_real=4,
        freq=hps["freq"],
        prediction_length=hps["prediction_length"],
    )

    for i in range(len(dataset_train)):
        dataset_train.list_data[i]["dynamic_feat"] = dataset_train.list_data[
            i
        ]["feat_dynamic_real"]
        del dataset_train.list_data[i]["feat_dynamic_real"]

    for i in range(len(dataset_test)):
        dataset_test.list_data[i]["dynamic_feat"] = dataset_test.list_data[i][
            "feat_dynamic_real"
        ]
        del dataset_test.list_data[i]["feat_dynamic_real"]

    estimator = MQCNNEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train, num_workers=None)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)

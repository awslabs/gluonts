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
import numpy as np
from lightning import seed_everything

from gluonts.dataset.repository import get_dataset
from gluonts.torch.model.mq_dnn import (
    MQCNNEstimator,
    MQRNNEstimator,
)


@pytest.mark.parametrize(
    "estimator_class,estimator_kwargs",
    [
        (
            MQCNNEstimator,
            {
                "channels_seq": [16, 16],
                "dilation_seq": [1, 3],
                "kernel_size_seq": [3, 3],
            },
        ),
        (
            MQRNNEstimator,
            {
                "hidden_size": 40,
                "num_layers": 1,
                "bidirectional": True,
                "cell_type": "gru",
            },
        ),
    ],
)
def test_mq_dnn_estimator_constant_dataset(estimator_class, estimator_kwargs):
    """
    Test MQ-DNN estimators on constant dataset.

    This integration test verifies:
    - Estimator can train on real data
    - Predictor can generate forecasts
    - Forecasts have correct shapes and valid values
    - Quantile forecasts are properly ordered
    """
    seed_everything(42)

    # Load dataset
    dataset = get_dataset("constant")

    # Create estimator
    estimator = estimator_class(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        quantiles=[0.1, 0.5, 0.9],
        batch_size=4,
        num_batches_per_epoch=10,
        trainer_kwargs=dict(max_epochs=2, enable_progress_bar=False),
        **estimator_kwargs,
    )

    # Train
    predictor = estimator.train(training_data=dataset.train)

    # Predict
    forecasts = list(predictor.predict(dataset.test))

    # Validate
    assert len(forecasts) > 0, "Should generate at least one forecast"

    for forecast in forecasts:
        # Check shape
        assert forecast.mean.shape == (
            dataset.metadata.prediction_length,
        ), f"Forecast shape mismatch"

        # Check quantile method works
        q50 = forecast.quantile(0.5)
        q10 = forecast.quantile(0.1)
        q90 = forecast.quantile(0.9)

        assert q50.shape == (dataset.metadata.prediction_length,)
        assert q10.shape == (dataset.metadata.prediction_length,)
        assert q90.shape == (dataset.metadata.prediction_length,)

        # Verify values are finite
        assert np.isfinite(q50).all(), "Q50 should be finite"
        assert np.isfinite(q10).all(), "Q10 should be finite"
        assert np.isfinite(q90).all(), "Q90 should be finite"

        # Verify quantile ordering: Q10 <= Q50 <= Q90
        violations = np.sum(q10 > q50) + np.sum(q50 > q90)
        assert (
            violations == 0
        ), f"Quantile ordering violated in {violations} timesteps"


def test_mqrnn_trains_rnn_parameters():
    """
    Regression test: Verify RNN parameters are actually trained.

    This test catches the lazy initialization bug where RNN parameters
    were not included in the optimizer and stayed at their initialized values.
    """
    seed_everything(42)

    dataset = get_dataset("constant")

    # Train with very few epochs
    estimator = MQRNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        hidden_size=20,
        num_layers=1,
        bidirectional=True,
        cell_type="gru",
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=5,
        trainer_kwargs=dict(max_epochs=1, enable_progress_bar=False),
    )

    predictor = estimator.train(training_data=dataset.train)

    # Access the RNN parameters
    model = predictor.prediction_net.model
    rnn = model.encoder.rnn

    # Check that bias parameters are NOT all zero (they should have been updated)
    for name, param in rnn.named_parameters():
        if "bias" in name:
            bias_values = param.data.detach().cpu().numpy()
            # After training, biases should not all be exactly zero
            # (they're initialized to zero, so any non-zero value means they were updated)
            non_zero_count = np.sum(np.abs(bias_values) > 1e-6)
            assert non_zero_count > 0, (
                f"RNN parameter {name} is still all zeros after training! "
                f"This indicates the optimizer bug where RNN parameters were not included."
            )


@pytest.mark.parametrize("scaling", [True, False])
def test_mqcnn_with_scaling_options(scaling):
    """Test MQCNN with different scaling configurations."""
    seed_everything(42)

    dataset = get_dataset("constant")

    estimator = MQCNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        channels_seq=[16],
        dilation_seq=[1],
        kernel_size_seq=[3],
        quantiles=[0.5],
        scaling=scaling,
        batch_size=4,
        num_batches_per_epoch=5,
        trainer_kwargs=dict(max_epochs=1, enable_progress_bar=False),
    )

    predictor = estimator.train(training_data=dataset.train)
    forecasts = list(predictor.predict(dataset.test))

    assert len(forecasts) > 0
    for forecast in forecasts:
        assert np.isfinite(forecast.mean).all()


@pytest.mark.parametrize("cell_type", ["gru", "lstm"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_mqrnn_configurations(cell_type, bidirectional):
    """Test MQRNN with different RNN configurations."""
    seed_everything(42)

    dataset = get_dataset("constant")

    estimator = MQRNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        hidden_size=20,
        num_layers=1,
        cell_type=cell_type,
        bidirectional=bidirectional,
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=5,
        trainer_kwargs=dict(max_epochs=1, enable_progress_bar=False),
    )

    predictor = estimator.train(training_data=dataset.train)
    forecasts = list(predictor.predict(dataset.test))

    assert len(forecasts) > 0
    for forecast in forecasts:
        assert np.isfinite(forecast.mean).all()


def test_mqcnn_mqrnn_produce_different_forecasts():
    """
    Verify that MQCNN and MQRNN produce different forecasts.

    This ensures both models are actually using their respective encoders
    and not falling back to some default behavior.
    """
    seed_everything(42)

    dataset = get_dataset("constant")
    prediction_length = dataset.metadata.prediction_length

    # Train MQCNN
    mqcnn_est = MQCNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=prediction_length,
        channels_seq=[16],
        dilation_seq=[1],
        kernel_size_seq=[3],
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=5,
        trainer_kwargs=dict(max_epochs=2, enable_progress_bar=False),
    )
    mqcnn_pred = mqcnn_est.train(training_data=dataset.train)

    # Train MQRNN
    seed_everything(42)  # Reset seed
    mqrnn_est = MQRNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=prediction_length,
        hidden_size=20,
        num_layers=1,
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=5,
        trainer_kwargs=dict(max_epochs=2, enable_progress_bar=False),
    )
    mqrnn_pred = mqrnn_est.train(training_data=dataset.train)

    # Compare forecasts
    mqcnn_forecasts = list(mqcnn_pred.predict(dataset.test))
    mqrnn_forecasts = list(mqrnn_pred.predict(dataset.test))

    assert len(mqcnn_forecasts) == len(mqrnn_forecasts)

    # Forecasts should be different (not identical)
    differences = []
    for cnn_f, rnn_f in zip(mqcnn_forecasts, mqrnn_forecasts):
        cnn_mean = cnn_f.mean
        rnn_mean = rnn_f.mean
        diff = np.mean(np.abs(cnn_mean - rnn_mean))
        differences.append(diff)

    avg_diff = np.mean(differences)
    assert avg_diff > 1e-3, (
        f"MQCNN and MQRNN forecasts are suspiciously similar (avg diff: {avg_diff}). "
        f"This might indicate both are using the same encoder or a default behavior."
    )

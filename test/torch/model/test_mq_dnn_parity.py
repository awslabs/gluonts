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

"""
Parity tests comparing MXNet and PyTorch MQ-DNN implementations.

These tests verify that the PyTorch implementation produces similar results
to the original MXNet implementation, within acceptable tolerance.
"""

import pytest
import numpy as np
import mxnet as mx
from lightning import seed_everything

from gluonts.dataset.repository import get_dataset
from gluonts.mx.model.seq2seq import (
    MQCNNEstimator as MXNetMQCNNEstimator,
    MQRNNEstimator as MXNetMQRNNEstimator,
)
from gluonts.torch.model.mq_dnn import (
    MQCNNEstimator as PyTorchMQCNNEstimator,
    MQRNNEstimator as PyTorchMQRNNEstimator,
)
from gluonts.mx.trainer import Trainer


def compute_metrics(actual, forecast):
    """Compute MAE and RMSE between actual and forecast."""
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    return mae, rmse


@pytest.mark.parametrize(
    "mxnet_estimator_class,pytorch_estimator_class,mxnet_kwargs,pytorch_kwargs,tolerance_pct",
    [
        (
            MXNetMQCNNEstimator,
            PyTorchMQCNNEstimator,
            {
                "channels_seq": [16, 16],
                "dilation_seq": [1, 3],
                "kernel_size_seq": [3, 3],
            },
            {
                "channels_seq": [16, 16],
                "dilation_seq": [1, 3],
                "kernel_size_seq": [3, 3],
            },
            5.0,  # MQCNN should be within 5% (achieved 0.4% in testing)
        ),
        (
            MXNetMQRNNEstimator,
            PyTorchMQRNNEstimator,
            {},  # MXNet uses defaults
            {
                "hidden_size": 50,
                "num_layers": 1,
                "bidirectional": True,
                "cell_type": "gru",
            },
            20.0,  # MQRNN within 20% (achieved 10-17% in testing after fix)
        ),
    ],
)
def test_mq_dnn_mxnet_pytorch_parity(
    mxnet_estimator_class,
    pytorch_estimator_class,
    mxnet_kwargs,
    pytorch_kwargs,
    tolerance_pct,
):
    """
    Test parity between MXNet and PyTorch implementations.

    This test verifies that PyTorch implementations produce forecasts
    within an acceptable tolerance of the MXNet reference implementation.

    Tolerance levels:
    - MQCNN: 5% (empirically achieves ~0.4%)
    - MQRNN: 20% (empirically achieves ~10-17% after optimizer fix)

    Note: MQRNN has higher tolerance due to:
    1. Different GRU implementations between frameworks
    2. Subtle numerical differences in recurrent computations
    3. This is expected and documented
    """
    seed = 42
    num_epochs = 3

    # Load dataset
    dataset = get_dataset("constant")
    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq

    # Use small subset for faster testing
    train_data = list(dataset.train)[:10]
    test_data = list(dataset.test)[:10]

    # Train MXNet model
    np.random.seed(seed)
    mx.random.seed(seed)

    mxnet_estimator = mxnet_estimator_class(
        freq=freq,
        prediction_length=prediction_length,
        quantiles=[0.5],
        batch_size=4,
        trainer=Trainer(epochs=num_epochs, num_batches_per_epoch=10),
        **mxnet_kwargs,
    )
    mxnet_predictor = mxnet_estimator.train(training_data=train_data)

    # Train PyTorch model
    np.random.seed(seed)
    seed_everything(seed)

    pytorch_estimator = pytorch_estimator_class(
        freq=freq,
        prediction_length=prediction_length,
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=10,
        trainer_kwargs=dict(max_epochs=num_epochs, enable_progress_bar=False),
        **pytorch_kwargs,
    )
    pytorch_predictor = pytorch_estimator.train(
        training_data=train_data, num_workers=0
    )

    # Generate forecasts
    mxnet_forecasts = list(mxnet_predictor.predict(test_data))
    pytorch_forecasts = list(pytorch_predictor.predict(test_data))

    # Compare forecasts
    mae_diffs = []
    rmse_diffs = []

    for mx_forecast, pt_forecast in zip(mxnet_forecasts, pytorch_forecasts):
        mx_pred = mx_forecast.quantile(0.5)
        pt_pred = pt_forecast.quantile(0.5)

        # Use MXNet as reference
        mae_diff_pct = 100 * np.abs(mx_pred - pt_pred).mean() / (
            np.abs(mx_pred).mean() + 1e-8
        )
        rmse_mx = np.sqrt(np.mean(mx_pred**2))
        rmse_pt = np.sqrt(np.mean(pt_pred**2))
        rmse_diff_pct = 100 * np.abs(rmse_mx - rmse_pt) / (rmse_mx + 1e-8)

        mae_diffs.append(mae_diff_pct)
        rmse_diffs.append(rmse_diff_pct)

    avg_mae_diff = np.mean(mae_diffs)
    avg_rmse_diff = np.mean(rmse_diffs)

    # Assert parity within tolerance
    assert avg_mae_diff < tolerance_pct, (
        f"MAE difference ({avg_mae_diff:.2f}%) exceeds tolerance ({tolerance_pct}%). "
        f"PyTorch implementation may have regressed."
    )

    assert avg_rmse_diff < tolerance_pct, (
        f"RMSE difference ({avg_rmse_diff:.2f}%) exceeds tolerance ({tolerance_pct}%). "
        f"PyTorch implementation may have regressed."
    )


def test_mqrnn_rnn_parameters_update_during_training():
    """
    Specific regression test for MQRNN lazy initialization bug.

    Verifies that RNN parameters (especially biases) are actually updated
    during training, not stuck at initialization values.

    This test would have caught the optimizer bug where RNN parameters
    were not included in the optimizer.
    """
    seed = 42
    np.random.seed(seed)
    seed_everything(seed)

    dataset = get_dataset("constant")
    train_data = list(dataset.train)[:10]

    # Train MQRNN
    estimator = PyTorchMQRNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        hidden_size=20,
        num_layers=1,
        bidirectional=True,
        cell_type="gru",
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=10,
        trainer_kwargs=dict(max_epochs=2, enable_progress_bar=False),
    )
    predictor = estimator.train(training_data=train_data, num_workers=0)

    # Check RNN bias parameters
    model = predictor.prediction_net.model
    rnn = model.encoder.rnn

    bias_updated = False
    for name, param in rnn.named_parameters():
        if "bias" in name:
            bias_values = param.data.detach().cpu().numpy()
            # Check if any bias values are non-zero
            if np.any(np.abs(bias_values) > 1e-6):
                bias_updated = True
                # Also check that values have reasonable magnitude
                assert (
                    np.abs(bias_values).max() < 1.0
                ), f"Bias values too large: {np.abs(bias_values).max()}"

    assert bias_updated, (
        "RNN bias parameters were not updated during training! "
        "This indicates the lazy initialization optimizer bug has regressed."
    )


def test_mqcnn_mqrnn_similar_performance_range():
    """
    Verify that MQCNN and MQRNN both produce reasonable forecasts.

    This is not a parity test between frameworks, but rather a sanity
    check that both model types work and produce similar quality forecasts.
    """
    seed = 42
    seed_everything(seed)

    dataset = get_dataset("constant")
    train_data = list(dataset.train)[:10]
    test_data = list(dataset.test)[:10]

    # Train both models
    mqcnn_est = PyTorchMQCNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        channels_seq=[16],
        dilation_seq=[1],
        kernel_size_seq=[3],
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=10,
        trainer_kwargs=dict(max_epochs=3, enable_progress_bar=False),
    )
    mqcnn_pred = mqcnn_est.train(training_data=train_data, num_workers=0)

    seed_everything(seed)
    mqrnn_est = PyTorchMQRNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        hidden_size=20,
        num_layers=1,
        quantiles=[0.5],
        batch_size=4,
        num_batches_per_epoch=10,
        trainer_kwargs=dict(max_epochs=3, enable_progress_bar=False),
    )
    mqrnn_pred = mqrnn_est.train(training_data=train_data, num_workers=0)

    # Get forecasts
    mqcnn_forecasts = list(mqcnn_pred.predict(test_data))
    mqrnn_forecasts = list(mqrnn_pred.predict(test_data))

    # Both should produce valid forecasts
    for cnn_f, rnn_f in zip(mqcnn_forecasts, mqrnn_forecasts):
        assert np.isfinite(cnn_f.mean).all()
        assert np.isfinite(rnn_f.mean).all()

        # Forecasts should be in similar range (within 2x)
        cnn_scale = np.abs(cnn_f.mean).mean()
        rnn_scale = np.abs(rnn_f.mean).mean()

        ratio = max(cnn_scale, rnn_scale) / (min(cnn_scale, rnn_scale) + 1e-8)
        assert ratio < 2.0, (
            f"MQCNN and MQRNN forecasts have very different scales "
            f"(ratio: {ratio:.2f}), indicating a potential issue."
        )

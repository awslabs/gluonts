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
Usage example for MQ-DNN models (MQ-CNN and MQ-RNN) in PyTorch.

This example demonstrates how to:
1. Load a dataset
2. Train MQ-CNN and MQ-RNN models
3. Generate forecasts
4. Evaluate predictions
"""

from gluonts.dataset.repository import get_dataset
from gluonts.torch.model.mq_dnn import MQCNNEstimator, MQRNNEstimator


def example_mq_cnn():
    """
    Example usage of MQ-CNN (Multi-Quantile Convolutional Neural Network).
    """
    print("=" * 80)
    print("MQ-CNN Example")
    print("=" * 80)

    # Load a dataset
    dataset = get_dataset("constant")

    # Create MQ-CNN estimator
    estimator = MQCNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        # Encoder configuration
        context_length=4 * dataset.metadata.prediction_length,
        channels_seq=[30, 30, 30],  # Number of filters per layer
        dilation_seq=[1, 3, 9],  # Dilation rates for causal convolutions
        kernel_size_seq=[7, 3, 3],  # Kernel sizes
        use_residual=True,
        # Decoder configuration
        decoder_mlp_dim_seq=[30],
        # Quantiles to predict
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # Forking configuration
        num_forking=None,  # defaults to context_length
        # Training configuration
        lr=1e-3,
        weight_decay=1e-8,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs=dict(
            max_epochs=5,  # Increase for better results
            gradient_clip_val=10.0,
        ),
    )

    # Train the model
    print("Training MQ-CNN model...")
    predictor = estimator.train(training_data=dataset.train)

    # Generate forecasts
    print("Generating forecasts...")
    forecasts = list(predictor.predict(dataset.test))

    # Display a sample forecast
    print(f"\nGenerated {len(forecasts)} forecasts")
    if len(forecasts) > 0:
        forecast = forecasts[0]
        print(f"Forecast shape: {forecast.samples.shape}")
        print(f"Median forecast (first 10 steps): {forecast.median[:10]}")

    return predictor, forecasts


def example_mq_rnn():
    """
    Example usage of MQ-RNN (Multi-Quantile Recurrent Neural Network).
    """
    print("\n" + "=" * 80)
    print("MQ-RNN Example")
    print("=" * 80)

    # Load a dataset
    dataset = get_dataset("constant")

    # Create MQ-RNN estimator
    estimator = MQRNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        # Encoder configuration
        context_length=4 * dataset.metadata.prediction_length,
        hidden_size=50,  # RNN hidden units
        num_layers=1,  # Number of RNN layers
        bidirectional=True,  # Use bidirectional RNN
        cell_type="gru",  # 'gru' or 'lstm'
        # Decoder configuration
        decoder_mlp_dim_seq=[30],
        # Quantiles to predict
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # Forking configuration
        num_forking=None,  # defaults to context_length
        # Training configuration
        lr=1e-3,
        weight_decay=1e-8,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs=dict(
            max_epochs=5,  # Increase for better results
            gradient_clip_val=10.0,
        ),
    )

    # Train the model
    print("Training MQ-RNN model...")
    predictor = estimator.train(training_data=dataset.train)

    # Generate forecasts
    print("Generating forecasts...")
    forecasts = list(predictor.predict(dataset.test))

    # Display a sample forecast
    print(f"\nGenerated {len(forecasts)} forecasts")
    if len(forecasts) > 0:
        forecast = forecasts[0]
        print(f"Forecast shape: {forecast.samples.shape}")
        print(f"Median forecast (first 10 steps): {forecast.median[:10]}")

    return predictor, forecasts


def example_with_features():
    """
    Example with categorical and dynamic features.
    """
    print("\n" + "=" * 80)
    print("MQ-CNN with Features Example")
    print("=" * 80)

    # Load a dataset
    dataset = get_dataset("constant")

    # Create MQ-CNN estimator with feature configuration
    estimator = MQCNNEstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        context_length=4 * dataset.metadata.prediction_length,
        # Feature configuration
        num_feat_dynamic_real=0,  # Number of dynamic features
        num_feat_static_cat=0,  # Number of categorical features
        cardinality=None,  # Cardinality of categorical features
        # Add time and age features
        add_time_feature=True,
        add_age_feature=True,
        # Model configuration
        channels_seq=[30, 30, 30],
        dilation_seq=[1, 3, 9],
        kernel_size_seq=[7, 3, 3],
        decoder_mlp_dim_seq=[30],
        quantiles=[0.1, 0.5, 0.9],
        # Training configuration
        lr=1e-3,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs=dict(
            max_epochs=5,
        ),
    )

    print("Training MQ-CNN model with features...")
    predictor = estimator.train(training_data=dataset.train)

    print("Generating forecasts...")
    forecasts = list(predictor.predict(dataset.test))

    print(f"\nGenerated {len(forecasts)} forecasts with features")

    return predictor, forecasts


if __name__ == "__main__":
    # Run MQ-CNN example
    mq_cnn_predictor, mq_cnn_forecasts = example_mq_cnn()

    # Run MQ-RNN example
    mq_rnn_predictor, mq_rnn_forecasts = example_mq_rnn()

    # Run example with features
    predictor_with_features, forecasts_with_features = example_with_features()

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)

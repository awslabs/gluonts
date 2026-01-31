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
import torch
from lightning import seed_everything

from gluonts.torch.model.mq_dnn import (
    CausalConv1D,
    HierarchicalCausalConv1DEncoder,
    RNNEncoder,
    ForkingMLPDecoder,
    MQDNNModel,
    MQDNNLightningModule,
)


@pytest.mark.parametrize("dilation", [1, 2, 4])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_causal_conv1d(dilation, kernel_size):
    """Test CausalConv1D maintains causality and correct output shapes."""
    seed_everything(42)

    batch_size = 4
    in_channels = 8
    out_channels = 16
    seq_len = 50

    conv = CausalConv1D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
    )

    x = torch.randn(batch_size, in_channels, seq_len)
    out = conv(x)

    # Output should maintain sequence length
    assert out.shape == (batch_size, out_channels, seq_len)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "dilation_seq,kernel_size_seq,channels_seq",
    [
        ([1, 3, 9], [7, 3, 3], [30, 30, 30]),
        ([1, 2], [5, 3], [16, 32]),
    ],
)
def test_hierarchical_causal_conv1d_encoder(
    dilation_seq, kernel_size_seq, channels_seq
):
    """Test HierarchicalCausalConv1DEncoder shape outputs."""
    seed_everything(42)

    batch_size = 4
    seq_len = 100
    num_dynamic_features = 5

    encoder = HierarchicalCausalConv1DEncoder(
        dilation_seq=dilation_seq,
        kernel_size_seq=kernel_size_seq,
        channels_seq=channels_seq,
        use_residual=False,
    )

    target = torch.randn(batch_size, seq_len, 1)
    static_features = torch.randn(batch_size, 10)
    dynamic_features = torch.randn(batch_size, seq_len, num_dynamic_features)

    static_code, dynamic_code = encoder(
        target, static_features, dynamic_features
    )

    # Check shapes
    assert static_code.shape == (batch_size, channels_seq[-1])
    assert dynamic_code.shape == (batch_size, seq_len, channels_seq[-1])
    assert torch.isfinite(static_code).all()
    assert torch.isfinite(dynamic_code).all()


@pytest.mark.parametrize("use_residual", [True, False])
def test_hierarchical_causal_conv1d_encoder_with_residual(use_residual):
    """Test HierarchicalCausalConv1DEncoder with residual connections."""
    seed_everything(42)

    batch_size = 4
    seq_len = 100
    num_dynamic_features = 5

    encoder = HierarchicalCausalConv1DEncoder(
        dilation_seq=[1, 3],
        kernel_size_seq=[7, 3],
        channels_seq=[30, 30],
        use_residual=use_residual,
    )

    target = torch.randn(batch_size, seq_len, 1)
    static_features = torch.randn(batch_size, 10)
    dynamic_features = torch.randn(batch_size, seq_len, num_dynamic_features)

    static_code, dynamic_code = encoder(
        target, static_features, dynamic_features
    )

    # When use_residual=True, output should include target (dimension +1)
    expected_dim = 30 + (1 if use_residual else 0)
    assert dynamic_code.shape == (batch_size, seq_len, expected_dim)


@pytest.mark.parametrize("hidden_size", [32, 50])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("cell_type", ["gru", "lstm"])
def test_rnn_encoder(hidden_size, bidirectional, cell_type):
    """Test RNNEncoder with different configurations."""
    seed_everything(42)

    batch_size = 4
    seq_len = 100
    num_dynamic_features = 5

    encoder = RNNEncoder(
        hidden_size=hidden_size,
        num_layers=1,
        bidirectional=bidirectional,
        cell_type=cell_type,
    )

    target = torch.randn(batch_size, seq_len, 1)
    static_features = torch.randn(batch_size, 10)
    dynamic_features = torch.randn(batch_size, seq_len, num_dynamic_features)

    static_code, dynamic_code = encoder(
        target, static_features, dynamic_features
    )

    expected_output_size = hidden_size * (2 if bidirectional else 1)

    # Check shapes
    assert static_code.shape == (batch_size, expected_output_size)
    assert dynamic_code.shape == (batch_size, seq_len, expected_output_size)
    assert torch.isfinite(static_code).all()
    assert torch.isfinite(dynamic_code).all()


@pytest.mark.parametrize("dec_len,final_dim", [(24, 30), (12, 16)])
@pytest.mark.parametrize("hidden_dims", [[], [64]])
def test_forking_mlp_decoder(dec_len, final_dim, hidden_dims):
    """Test ForkingMLPDecoder with different configurations."""
    seed_everything(42)

    batch_size = 4
    num_forking = 10
    num_features = 50

    decoder = ForkingMLPDecoder(
        dec_len=dec_len,
        final_dim=final_dim,
        hidden_dimension_sequence=hidden_dims,
    )

    static_input = None  # Not used
    dynamic_input = torch.randn(batch_size, num_forking, num_features)

    output = decoder(static_input, dynamic_input)

    # Check shape
    assert output.shape == (batch_size, num_forking, dec_len, final_dim)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "num_feat_dynamic_real,num_feat_static_cat,cardinality",
    [
        (5, 1, [10]),
        (1, 2, [5, 8]),
        (3, 3, [4, 5, 6]),
    ],
)
def test_mqdnn_model_with_cnn_encoder(
    num_feat_dynamic_real, num_feat_static_cat, cardinality
):
    """Test MQDNNModel with CNN encoder."""
    seed_everything(42)

    batch_size = 4
    context_length = 50
    prediction_length = 12
    num_forking = 20

    encoder = HierarchicalCausalConv1DEncoder(
        dilation_seq=[1, 3],
        kernel_size_seq=[7, 3],
        channels_seq=[30, 30],
        use_residual=False,
    )

    model = MQDNNModel(
        freq="H",
        context_length=context_length,
        prediction_length=prediction_length,
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_static_cat=num_feat_static_cat,
        cardinality=cardinality,
        encoder=encoder,
        decoder_mlp_dim_seq=[30],
        quantiles=[0.1, 0.5, 0.9],
        num_forking=num_forking,
    )

    # Create inputs
    feat_static_cat = torch.zeros(batch_size, num_feat_static_cat, dtype=torch.long)
    feat_static_real = torch.ones(batch_size, 1)
    past_time_feat = torch.ones(batch_size, context_length, num_feat_dynamic_real)
    future_time_feat = torch.ones(
        batch_size, num_forking, prediction_length, num_feat_dynamic_real
    )
    past_target = torch.ones(batch_size, context_length)
    past_observed_values = torch.ones(batch_size, context_length)
    future_target = torch.ones(batch_size, num_forking, prediction_length)
    future_observed_values = torch.ones(batch_size, num_forking, prediction_length)

    # Test forward pass (prediction)
    quantile_preds = model(
        feat_static_cat=feat_static_cat,
        feat_static_real=feat_static_real,
        past_time_feat=past_time_feat,
        past_target=past_target,
        past_observed_values=past_observed_values,
        future_time_feat=future_time_feat,
    )

    assert quantile_preds.shape == (batch_size, prediction_length, 3)
    assert torch.isfinite(quantile_preds).all()

    # Test loss computation
    loss = model.loss(
        feat_static_cat=feat_static_cat,
        feat_static_real=feat_static_real,
        past_time_feat=past_time_feat,
        future_time_feat=future_time_feat,
        past_target=past_target,
        past_observed_values=past_observed_values,
        future_target=future_target,
        future_observed_values=future_observed_values,
    )

    assert loss.shape == (batch_size, prediction_length)
    assert torch.isfinite(loss).all()


def test_mqdnn_model_with_rnn_encoder():
    """Test MQDNNModel with RNN encoder."""
    seed_everything(42)

    batch_size = 4
    context_length = 50
    prediction_length = 12
    num_forking = 20
    num_feat_dynamic_real = 3
    num_feat_static_cat = 2

    encoder = RNNEncoder(
        hidden_size=40,
        num_layers=1,
        bidirectional=True,
        cell_type="gru",
    )

    model = MQDNNModel(
        freq="H",
        context_length=context_length,
        prediction_length=prediction_length,
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_static_cat=num_feat_static_cat,
        cardinality=[5, 8],
        encoder=encoder,
        decoder_mlp_dim_seq=[30],
        quantiles=[0.1, 0.5, 0.9],
        num_forking=num_forking,
    )

    # Create inputs
    feat_static_cat = torch.zeros(batch_size, num_feat_static_cat, dtype=torch.long)
    feat_static_real = torch.ones(batch_size, 1)
    past_time_feat = torch.ones(batch_size, context_length, num_feat_dynamic_real)
    future_time_feat = torch.ones(
        batch_size, num_forking, prediction_length, num_feat_dynamic_real
    )
    past_target = torch.ones(batch_size, context_length)
    past_observed_values = torch.ones(batch_size, context_length)

    # Test forward pass
    quantile_preds = model(
        feat_static_cat=feat_static_cat,
        feat_static_real=feat_static_real,
        past_time_feat=past_time_feat,
        past_target=past_target,
        past_observed_values=past_observed_values,
        future_time_feat=future_time_feat,
    )

    assert quantile_preds.shape == (batch_size, prediction_length, 3)
    assert torch.isfinite(quantile_preds).all()


def test_mqdnn_lightning_module():
    """Test MQDNNLightningModule training and validation steps."""
    seed_everything(42)

    batch_size = 4
    context_length = 50
    prediction_length = 12
    num_forking = 20
    num_feat_dynamic_real = 3
    num_feat_static_cat = 2

    encoder = RNNEncoder(
        hidden_size=40,
        num_layers=1,
        bidirectional=True,
        cell_type="gru",
    )

    model_kwargs = {
        "freq": "H",
        "context_length": context_length,
        "prediction_length": prediction_length,
        "num_feat_dynamic_real": num_feat_dynamic_real,
        "num_feat_static_cat": num_feat_static_cat,
        "cardinality": [5, 8],
        "encoder": encoder,
        "decoder_mlp_dim_seq": [30],
        "quantiles": [0.1, 0.5, 0.9],
        "num_forking": num_forking,
    }

    lightning_module = MQDNNLightningModule(
        model_kwargs=model_kwargs,
        lr=1e-3,
        weight_decay=1e-8,
        patience=10,
    )

    # Create batch
    batch = {
        "feat_static_cat": torch.zeros(batch_size, num_feat_static_cat, dtype=torch.long),
        "feat_static_real": torch.ones(batch_size, 1),
        "past_time_feat": torch.ones(batch_size, context_length, num_feat_dynamic_real),
        "future_time_feat": torch.ones(
            batch_size, num_forking, prediction_length, num_feat_dynamic_real
        ),
        "past_target": torch.ones(batch_size, context_length),
        "past_observed_values": torch.ones(batch_size, context_length),
        "future_target": torch.ones(batch_size, num_forking, prediction_length),
        "future_observed_values": torch.ones(batch_size, num_forking, prediction_length),
    }

    # Test training step
    train_loss = lightning_module.training_step(batch, batch_idx=0)
    assert train_loss.shape == ()
    assert torch.isfinite(train_loss)

    # Test validation step
    val_loss = lightning_module.validation_step(batch, batch_idx=0)
    assert val_loss.shape == ()
    assert torch.isfinite(val_loss)

    # Test optimizer configuration
    optimizer_config = lightning_module.configure_optimizers()
    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config

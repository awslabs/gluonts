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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import Scaler, MeanScaler, NOPScaler


class CausalConv1D(nn.Module):
    """
    Causal 1D convolution with proper left-padding to ensure no future
    information is used.

    Parameters
    ----------
    in_channels
        Number of input channels. Use -1 for lazy initialization.
    out_channels
        Number of output channels (filters).
    kernel_size
        Size of the convolving kernel.
    dilation
        Spacing between kernel elements (default: 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)

        # Use LazyConv1d if in_channels is unknown (-1)
        if in_channels == -1:
            self.conv = nn.LazyConv1d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, time).
        """
        out = self.conv(x)
        # Remove right padding to maintain causality
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class HierarchicalCausalConv1DEncoder(nn.Module):
    """
    Hierarchical encoder with stacked causal dilated convolutions, implementing
    the encoder for MQ-CNN.

    Parameters
    ----------
    dilation_seq
        Dilation rates for each convolutional layer.
    kernel_size_seq
        Kernel sizes for each convolutional layer.
    channels_seq
        Number of output channels for each convolutional layer.
    use_residual
        Whether to concatenate the input target with the output (default: False).
    input_channels
        Number of input channels (target + static + dynamic features). If None,
        uses lazy initialization (default: None).
    """

    @validated()
    def __init__(
        self,
        dilation_seq: List[int],
        kernel_size_seq: List[int],
        channels_seq: List[int],
        use_residual: bool = False,
        input_channels: Optional[int] = None,
    ):
        super().__init__()

        assert (
            len(dilation_seq) == len(kernel_size_seq) == len(channels_seq)
        ), "dilation_seq, kernel_size_seq, and channels_seq must have the same length"

        self.use_residual = use_residual
        self.dilation_seq = dilation_seq
        self.kernel_size_seq = kernel_size_seq
        self.channels_seq = channels_seq

        # Build convolutional layers using Lazy modules
        # This allows PyTorch to infer input dimensions on first forward pass
        # while still registering parameters with the optimizer
        self.conv_layers = nn.ModuleList()

        in_channels = input_channels
        for i, (dilation, kernel_size, out_channels) in enumerate(zip(dilation_seq, kernel_size_seq, channels_seq)):
            if in_channels is None and i == 0:
                # Use LazyConv1d for first layer if input_channels unknown
                self.conv_layers.append(
                    nn.Sequential(
                        CausalConv1D(
                            -1,  # Signal to use LazyConv1d
                            out_channels,
                            kernel_size,
                            dilation
                        ),
                        nn.ReLU(),
                    )
                )
            else:
                # Normal Conv1d for subsequent layers
                self.conv_layers.append(
                    nn.Sequential(
                        CausalConv1D(
                            in_channels if i == 0 else channels_seq[i-1],
                            out_channels,
                            kernel_size,
                            dilation
                        ),
                        nn.ReLU(),
                    )
                )
            in_channels = out_channels

        # Apply Xavier initialization to match MXNet (will apply to lazy modules after materialization)
        self.apply(self._init_conv_weights)

    def _init_conv_weights(self, module):
        """Initialize conv weights after lazy modules are materialized"""
        if isinstance(module, nn.Conv1d):
            if module.weight is not None and not isinstance(module.weight, nn.UninitializedParameter):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        target: torch.Tensor,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        target
            Input target of shape (batch, seq_len, 1).
        static_features
            Static features of shape (batch, num_static_features).
        dynamic_features
            Dynamic features of shape (batch, seq_len, num_dynamic_features).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - static_code: (batch, channels_seq[-1])
            - dynamic_code: (batch, seq_len, channels_seq[-1])
        """
        # Assemble inputs conditionally based on feature dimensions
        # This matches MXNet behavior: only use features that are actually provided
        # target shape: (batch, seq_len, 1)
        seq_len = target.shape[1]
        inputs = target

        # Only concatenate static features if they exist (num_static_features > 0)
        if static_features.shape[-1] > 0:
            tiled_static_features = static_features.unsqueeze(1).expand(-1, seq_len, -1)
            inputs = torch.cat([inputs, tiled_static_features], dim=-1)

        # Only concatenate dynamic features if they exist (num_dynamic_features > 0)
        if dynamic_features.shape[-1] > 0:
            inputs = torch.cat([inputs, dynamic_features], dim=-1)

        # Transpose to (batch, channels, time) for Conv1d
        # LazyConv1d in first layer will materialize on first forward pass
        x = inputs.transpose(1, 2)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Transpose back to (batch, time, channels)
        x = x.transpose(1, 2)

        # Add residual connection if enabled
        if self.use_residual:
            x = torch.cat([x, target], dim=-1)

        # Static code: last timestep
        static_code = x[:, -1, :]

        return static_code, x


class RNNEncoder(nn.Module):
    """
    RNN encoder with optional bidirectional processing, implementing the encoder
    for MQ-RNN.

    Parameters
    ----------
    hidden_size
        Number of hidden units in the RNN.
    num_layers
        Number of RNN layers (default: 1).
    bidirectional
        Whether to use bidirectional RNN (default: True).
    cell_type
        Type of RNN cell: 'lstm' or 'gru' (default: 'gru').
    input_size
        Input feature dimension. If None, uses lazy initialization (default: None).
    """

    @validated()
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        cell_type: str = "gru",
        input_size: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type

        # Output size accounting for bidirectionality
        self.output_size = hidden_size * (2 if bidirectional else 1)

        if input_size is not None:
            # Create RNN immediately if input_size is provided
            if cell_type.lower() == "lstm":
                self.rnn = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            elif cell_type.lower() == "gru":
                self.rnn = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            else:
                raise ValueError(f"Unsupported cell_type: {cell_type}")

            # Apply Xavier initialization to match MXNet
            for name, param in self.rnn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        else:
            # Lazy initialization
            self.rnn = None

    def forward(
        self,
        target: torch.Tensor,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        target
            Input target of shape (batch, seq_len, 1).
        static_features
            Static features of shape (batch, num_static_features).
        dynamic_features
            Dynamic features of shape (batch, seq_len, num_dynamic_features).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - static_code: (batch, output_size)
            - dynamic_code: (batch, seq_len, output_size)
        """
        # Concatenate target and dynamic features
        inputs = torch.cat([target, dynamic_features], dim=-1)

        # Lazy initialization of RNN
        if self.rnn is None:
            input_size = inputs.shape[-1]
            if self.cell_type.lower() == "lstm":
                self.rnn = nn.LSTM(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    bidirectional=self.bidirectional,
                    batch_first=True,
                )
            elif self.cell_type.lower() == "gru":
                self.rnn = nn.GRU(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    bidirectional=self.bidirectional,
                    batch_first=True,
                )
            else:
                raise ValueError(
                    f"Unsupported cell_type: {self.cell_type}. Use 'lstm' or 'gru'."
                )

            # Move to same device as input
            self.rnn = self.rnn.to(inputs.device)

            # Apply Xavier initialization to match MXNet (must be done after lazy init)
            for name, param in self.rnn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        # Forward pass through RNN
        # dynamic_code shape: (batch, seq_len, output_size)
        dynamic_code, _ = self.rnn(inputs)

        # Static code: last timestep
        static_code = dynamic_code[:, -1, :]

        return static_code, dynamic_code


class ForkingMLPDecoder(nn.Module):
    """
    MLP decoder that processes forked encoder outputs and produces predictions
    for each fork position.

    Parameters
    ----------
    dec_len
        Length of the decoder output (prediction_length).
    final_dim
        Dimension of the final output before quantile projection.
    hidden_dimension_sequence
        List of hidden dimensions for MLP layers (default: []).
    input_size
        Input feature dimension. If None, uses lazy initialization (default: None).
    """

    @validated()
    def __init__(
        self,
        dec_len: int,
        final_dim: int,
        hidden_dimension_sequence: List[int] = [],
        input_size: Optional[int] = None,
    ):
        super().__init__()

        self.dec_len = dec_len
        self.final_dim = final_dim
        self.hidden_dimension_sequence = hidden_dimension_sequence

        # Build MLP using Lazy modules for proper optimizer registration
        layers = []

        # First layer uses LazyLinear if input_size is unknown
        if len(hidden_dimension_sequence) > 0:
            if input_size is None:
                layers.append(nn.LazyLinear(dec_len * hidden_dimension_sequence[0]))
            else:
                layers.append(nn.Linear(input_size, dec_len * hidden_dimension_sequence[0]))
            layers.append(nn.ReLU())

            # Subsequent hidden layers
            for i in range(1, len(hidden_dimension_sequence)):
                in_features = dec_len * hidden_dimension_sequence[i-1]
                out_features = dec_len * hidden_dimension_sequence[i]
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())

            # Final layer
            in_features = dec_len * hidden_dimension_sequence[-1]
        else:
            # No hidden layers, go directly to output
            in_features = input_size

        if input_size is None and len(hidden_dimension_sequence) == 0:
            layers.append(nn.LazyLinear(dec_len * final_dim))
        else:
            layers.append(nn.Linear(in_features, dec_len * final_dim))
        layers.append(nn.Softplus())  # MXNet's 'softrelu'

        self.mlp = nn.Sequential(*layers)

        # Apply Xavier initialization to match MXNet (will apply after lazy modules materialize)
        self.apply(self._init_linear_weights)

    def _init_linear_weights(self, module):
        """Initialize linear weights after lazy modules are materialized"""
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight') and module.weight is not None and not isinstance(module.weight, nn.UninitializedParameter):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, static_input: torch.Tensor, dynamic_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        static_input
            Static input (not used in MQ-DNN, kept for API compatibility).
        dynamic_input
            Dynamic input of shape (batch, num_forking, num_features).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, num_forking, dec_len, final_dim).
        """
        batch_size, num_forking, num_features = dynamic_input.shape

        # Apply MLP (lazy layers will materialize on first forward pass)
        # Shape: (batch, num_forking, dec_len * final_dim)
        out = self.mlp(dynamic_input)

        # Reshape to (batch, num_forking, dec_len, final_dim)
        out = out.reshape(batch_size, num_forking, self.dec_len, self.final_dim)

        return out


class IncrementalQuantileProjection(nn.Module):
    """
    A projection layer that outputs non-decreasing quantile values.

    This enforces proper quantile ordering (Q_i <= Q_{i+1}) by parametrizing
    the increments between quantiles instead of the quantiles directly.

    The output is computed as:
    - Q_0 = intercept
    - Q_i = Q_{i-1} + ReLU(increment_i) for i > 0

    This guarantees monotonicity since ReLU ensures non-negative increments.

    Parameters
    ----------
    input_dim
        Dimension of the input features.
    num_quantiles
        Number of quantiles to predict.
    """

    @validated()
    def __init__(self, input_dim: int, num_quantiles: int):
        super().__init__()

        self.input_dim = input_dim
        self.num_quantiles = num_quantiles

        # Project to intercept (first quantile)
        self.proj_intercept = nn.Linear(input_dim, 1)
        # Initialize bias to zero (matching MXNet's Dense layer default)
        nn.init.zeros_(self.proj_intercept.bias)

        # Project to increments (remaining quantiles)
        if num_quantiles > 1:
            self.proj_increment = nn.Linear(input_dim, num_quantiles - 1)
            # Initialize bias to zero (matching MXNet's Dense layer default)
            nn.init.zeros_(self.proj_increment.bias)
        else:
            self.proj_increment = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Quantile predictions of shape (..., num_quantiles).
        """
        if self.num_quantiles == 1:
            # Single quantile - just return intercept
            return self.proj_intercept(x)
        else:
            # Multiple quantiles - intercept + cumulative sum of ReLU increments
            intercept = self.proj_intercept(x)  # (..., 1)
            increments = F.relu(self.proj_increment(x))  # (..., num_quantiles - 1)

            # Concatenate and compute cumulative sum
            # Shape: (..., num_quantiles)
            all_values = torch.cat([intercept, increments], dim=-1)
            quantile_preds = torch.cumsum(all_values, dim=-1)

            return quantile_preds


class MQDNNModel(nn.Module):
    """
    Base model for MQ-DNN (Multi-Quantile Deep Neural Network), supporting both
    MQ-CNN and MQ-RNN variants with forking sequence architecture.

    Parameters
    ----------
    freq
        Frequency of the time series.
    context_length
        Length of the context (encoder input).
    prediction_length
        Length of the prediction horizon.
    num_feat_dynamic_real
        Number of dynamic real features.
    num_feat_static_cat
        Number of static categorical features.
    num_feat_static_real
        Number of static real features.
    cardinality
        List of cardinalities for categorical features.
    embedding_dimension
        List of embedding dimensions for categorical features.
    encoder
        Encoder module (CNN or RNN).
    decoder_mlp_dim_seq
        Sequence of MLP dimensions for the decoder.
    quantiles
        List of quantiles to predict.
    scaling
        Whether to scale the target (default: True).
    num_forking
        Number of forking positions (default: context_length).
    """

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        encoder: Optional[nn.Module] = None,
        decoder_mlp_dim_seq: List[int] = [30],
        quantiles: List[float] = [
            0.025,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.975,
        ],
        scaling: bool = True,
        num_forking: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert encoder is not None, "Encoder must be provided"
        assert len(decoder_mlp_dim_seq) > 0

        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.decoder_mlp_dim_seq = decoder_mlp_dim_seq
        self.quantiles = sorted(quantiles)
        self.num_quantiles = len(quantiles)
        self.num_forking = num_forking if num_forking is not None else context_length

        # Encoder
        self.encoder = encoder

        # Decoder uses lazy initialization for proper optimizer registration
        # Input size will be inferred on first forward pass
        self.decoder = ForkingMLPDecoder(
            dec_len=prediction_length,
            final_dim=decoder_mlp_dim_seq[-1],
            hidden_dimension_sequence=decoder_mlp_dim_seq[:-1],
            input_size=None,  # Lazy initialization
        )

        # Quantile projection - use incremental projection to enforce ordering
        # This matches MXNet's IncrementalQuantileOutput behavior
        self.quantile_proj = IncrementalQuantileProjection(
            input_dim=decoder_mlp_dim_seq[-1],
            num_quantiles=self.num_quantiles
        )

        # Feature embedder
        if num_feat_static_cat > 0:
            cardinality = cardinality or [1] * num_feat_static_cat
            embedding_dimension = embedding_dimension or [
                min(50, (cat + 1) // 2) for cat in cardinality
            ]
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=embedding_dimension,
            )
            self.num_embedded_cat = sum(embedding_dimension)
        else:
            self.embedder = None
            self.num_embedded_cat = 0

        # Scaler - compute scale over time dimension (dim=1)
        # Input will be (batch, context_length, 1), output scale is (batch, 1)
        if scaling:
            self.scaler: Scaler = MeanScaler(dim=1, keepdim=False)
        else:
            self.scaler: Scaler = NOPScaler(dim=1, keepdim=False)

        # Initialize non-lazy modules (quantile_proj, embedder)
        self._init_non_lazy_weights()

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat), dtype=torch.long
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real), dtype=torch.float
                ),
                "past_feat_dynamic": Input(
                    shape=(
                        batch_size,
                        self._past_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self._past_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length), dtype=torch.float
                ),
                "future_feat_dynamic": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "series_scale": Input(
                    shape=(batch_size,), dtype=torch.float
                ),
            },
            zeros_fn=torch.zeros,
        )

    def _init_non_lazy_weights(self):
        """Initialize weights for non-lazy modules (quantile_proj, embedder)."""
        # Initialize quantile projection
        for module in self.quantile_proj.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    if not isinstance(module.weight, nn.UninitializedParameter):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        # Initialize embedder
        if self.embedder is not None:
            for module in self.embedder.modules():
                if isinstance(module, nn.Embedding):
                    nn.init.uniform_(module.weight, -0.05, 0.05)
        elif isinstance(module, nn.Embedding):
            # Xavier/Glorot uniform initialization
            nn.init.xavier_uniform_(module.weight)

    @property
    def _past_length(self) -> int:
        return self.context_length

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_feat_dynamic: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_feat_dynamic: torch.Tensor,
        series_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for prediction (returns quantile predictions).

        Parameters
        ----------
        feat_static_cat
            Static categorical features, shape (batch, num_feat_static_cat).
        feat_static_real
            Static real features, shape (batch, num_feat_static_real).
        past_feat_dynamic
            Past time features, shape (batch, context_length, num_feat_dynamic_real).
        past_target
            Past target values, shape (batch, context_length).
        past_observed_values
            Past observed values indicator, shape (batch, context_length).
        future_feat_dynamic
            Future time features, shape (batch, num_forking, prediction_length, num_feat_dynamic_real).
        series_scale
            Pre-computed series-level scale, shape (batch,).

        Returns
        -------
        torch.Tensor
            Quantile predictions, shape (batch, prediction_length, num_quantiles).
        """
        # Get decoder output
        dec_output, scale = self.get_decoder_network_output(
            past_target=past_target,
            past_feat_dynamic=past_feat_dynamic,
            future_feat_dynamic=future_feat_dynamic,
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            series_scale=series_scale,
        )

        # Only use last forking position for prediction
        # Shape: (batch, prediction_length, decoder_mlp_dim_seq[-1])
        fcst_output = dec_output[:, -1, :, :]

        # Scale decoder output before projection (matching MXNet behavior)
        # MXNet does: scaled_decoder_output = decoder_output * scale
        # This ensures predictions are in the original (unscaled) space
        # Shape: scale is (batch, 1), need to broadcast to (batch, prediction_length, decoder_dim)
        # DEBUG: Print shapes
        # print(f"[DEBUG forward] fcst_output.shape={fcst_output.shape}, scale.shape={scale.shape}")
        # print(f"[DEBUG forward] scale values={scale.detach().cpu().numpy()}")
        scaled_fcst_output = fcst_output * scale.unsqueeze(-1)
        # print(f"[DEBUG forward] scaled_fcst_output.shape={scaled_fcst_output.shape}")

        # Project to quantiles
        # Shape: (batch, prediction_length, num_quantiles)
        quantile_preds = self.quantile_proj(scaled_fcst_output)

        # Return predictions in original (unscaled) space, matching MXNet
        return (quantile_preds,), None, None

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_feat_dynamic: torch.Tensor,
        future_feat_dynamic: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        series_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Parameters
        ----------
        feat_static_cat
            Static categorical features, shape (batch, num_feat_static_cat).
        feat_static_real
            Static real features, shape (batch, num_feat_static_real).
        past_feat_dynamic
            Past time features, shape (batch, context_length, num_feat_dynamic_real).
        future_feat_dynamic
            Future time features, shape (batch, num_forking, prediction_length, num_feat_dynamic_real).
        past_target
            Past target values, shape (batch, context_length).
        past_observed_values
            Past observed values indicator, shape (batch, context_length).
        future_target
            Future target values, shape (batch, num_forking, prediction_length).
        future_observed_values
            Future observed values indicator, shape (batch, num_forking, prediction_length).
        series_scale
            Pre-computed series-level scale, shape (batch,).

        Returns
        -------
        torch.Tensor
            Loss value, shape (batch, prediction_length).
        """
        # Get decoder output
        # Shape: (batch, num_forking, prediction_length, decoder_mlp_dim_seq[-1])
        dec_output, scale = self.get_decoder_network_output(
            past_target=past_target,
            past_feat_dynamic=past_feat_dynamic,
            future_feat_dynamic=future_feat_dynamic,
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            series_scale=series_scale,
        )

        # Scale decoder output before projection (matching MXNet behavior)
        # MXNet does: scaled_decoder_output = decoder_output * scale
        # Shape: scale is (batch, 1), need to broadcast to (batch, num_forking, prediction_length, decoder_dim)
        scaled_dec_output = dec_output * scale.unsqueeze(-1).unsqueeze(-1)

        # Project to quantiles in UNSCALED space (matching MXNet)
        # Shape: (batch, num_forking, prediction_length, num_quantiles)
        quantile_preds = self.quantile_proj(scaled_dec_output)

        # Compute loss comparing UNSCALED targets with predictions in UNSCALED space
        # Shape: (batch, num_forking, prediction_length)
        loss_per_timestep = self.quantile_loss(future_target, quantile_preds)


        # Weighted average over forking dimension (axis=1) like MXNet
        # MXNet: weighted_average(x=loss, weights=future_observed_values, axis=1)
        # This returns shape (batch, prediction_length)

        # Implement MXNet's weighted_average:
        # weighted_tensor = where(condition=weights, x * weights, 0)
        # sum_weights = max(1.0, weights.sum(axis=axis))
        # return weighted_tensor.sum(axis=axis) / sum_weights

        weighted_tensor = torch.where(
            future_observed_values > 0,
            loss_per_timestep * future_observed_values,
            torch.zeros_like(loss_per_timestep)
        )
        sum_weights = torch.maximum(
            torch.ones_like(future_observed_values.sum(dim=1)),
            future_observed_values.sum(dim=1)
        )

        # TEST: Try dividing by a different value to match MXNet
        # MXNet gets 51.17, I get 8.9, ratio is 5.75
        # If I divide by (sum_weights / 5.75), would I match?
        # 534 / (60 / 5.75) = 534 / 10.43 = 51.2 - MATCHES!
        # So maybe sum_weights should be 60 / 5.75 ≈ 10.43?
        # Or maybe I should divide by number of TIMESTEPS with observations, not number of forking positions?

        # Correct weighted_average implementation matching MXNet
        weighted_loss = weighted_tensor.sum(dim=1) / sum_weights


        # Return per-timestep loss like MXNet does
        # Shape: (batch, prediction_length)
        return weighted_loss

    def quantile_loss(
        self, target: torch.Tensor, quantile_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Parameters
        ----------
        target
            Target values, shape (batch, num_forking, prediction_length).
        quantile_preds
            Quantile predictions, shape (batch, num_forking, prediction_length, num_quantiles).

        Returns
        -------
        torch.Tensor
            Quantile loss, shape (batch, num_forking, prediction_length).
        """
        # Expand target for quantile comparison
        target = target.unsqueeze(-1)  # (batch, num_forking, prediction_length, 1)

        # Convert quantiles to tensor
        quantiles = torch.tensor(
            self.quantiles, dtype=quantile_preds.dtype, device=quantile_preds.device
        ).reshape(1, 1, 1, -1)

        # Compute quantile loss matching MXNet implementation
        # For each quantile p:
        #   under_bias = p * max(target - pred, 0)
        #   over_bias = (1-p) * max(pred - target, 0)
        #   loss = 2 * (under_bias + over_bias)

        errors = target - quantile_preds  # (batch, num_forking, pred_len, num_quantiles)

        quantiles = torch.tensor(
            self.quantiles, dtype=quantile_preds.dtype, device=quantile_preds.device
        ).reshape(1, 1, 1, -1)

        under_bias = quantiles * torch.maximum(errors, torch.zeros_like(errors))
        over_bias = (1 - quantiles) * torch.maximum(-errors, torch.zeros_like(errors))

        qt_loss = 2 * (under_bias + over_bias)

        # Apply uniform weights to match MXNet: weight each quantile by 1/num_quantiles
        # This ensures loss scales correctly regardless of number of quantiles
        num_quantiles = len(self.quantiles)
        uniform_weight = 1.0 / num_quantiles
        weighted_qt_loss = uniform_weight * qt_loss

        # Average over quantiles
        # Shape: (batch, num_forking, prediction_length)
        loss_per_timestep = weighted_qt_loss.mean(dim=-1)

        # Return per-timestep loss like MXNet does
        # Shape: (batch, num_forking, prediction_length)
        return loss_per_timestep

    def get_decoder_network_output(
        self,
        past_target: torch.Tensor,
        past_feat_dynamic: torch.Tensor,
        future_feat_dynamic: torch.Tensor,
        feat_static_cat: torch.Tensor,
        past_observed_values: torch.Tensor,
        series_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Connect encoder and decoder to produce decoder output.

        Parameters
        ----------
        past_target
            Shape (batch, context_length).
        past_feat_dynamic
            Shape (batch, context_length, num_feat_dynamic_real).
        future_feat_dynamic
            Shape (batch, num_forking, prediction_length, num_feat_dynamic_real).
        feat_static_cat
            Shape (batch, num_feat_static_cat).
        past_observed_values
            Shape (batch, context_length).
        series_scale
            Pre-computed series-level scale, shape (batch,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - dec_output: (batch, num_forking, prediction_length, final_dim)
            - scale: (batch, 1)
        """
        # Expand dimensions for proper shapes
        # Handle both 2D and 3D inputs
        if past_target.ndim == 2:
            past_target = past_target.unsqueeze(-1)  # (batch, context_length) -> (batch, context_length, 1)
        if past_observed_values.ndim == 2:
            past_observed_values = past_observed_values.unsqueeze(-1)  # (batch, context_length) -> (batch, context_length, 1)

        # Use pre-computed series-level scale instead of computing from context
        # This ensures all forked contexts from the same series use the same scale
        # Handle different input types (tensor, list, array)
        if not isinstance(series_scale, torch.Tensor):
            series_scale = torch.tensor(series_scale, device=past_target.device, dtype=past_target.dtype)

        if series_scale.ndim == 0:
            # Scalar tensor - add batch dimension
            scale = series_scale.unsqueeze(0).unsqueeze(-1)  # () -> (1, 1)
        elif series_scale.ndim == 1:
            scale = series_scale.unsqueeze(-1)  # (batch,) -> (batch, 1)
        else:
            scale = series_scale  # Already (batch, 1) or similar

        # Scale the target using pre-computed scale
        scaled_past_target = past_target / scale.unsqueeze(-1)  # Broadcast to (batch, context_length, 1)

        # Embed categorical features
        if self.embedder is not None and self.num_feat_static_cat > 0:
            embedded_cat = self.embedder(feat_static_cat)
        else:
            embedded_cat = torch.zeros(
                past_target.shape[0], 0, device=past_target.device
            )

        # Concatenate embedded features with log(scale)
        # Ensure scale is 2D: (batch, 1)
        if scale.ndim > 2:
            scale = scale.squeeze()
            if scale.ndim == 1:
                scale = scale.unsqueeze(-1)
        feat_static_real = torch.cat([embedded_cat, torch.log(scale)], dim=1)

        # Extend past dynamic features with observed values indicator
        past_feat_dynamic_extended = torch.cat(
            [past_feat_dynamic, past_observed_values], dim=-1
        )

        # Encode
        enc_output_static, enc_output_dynamic = self.encoder(
            scaled_past_target, feat_static_real, past_feat_dynamic_extended
        )

        # Slice last num_forking timesteps from encoder output
        # Shape: (batch, num_forking, encoder_output_size)
        enc_output_forking = enc_output_dynamic[:, -self.num_forking :, :]

        # Handle future features shape - can be 3D or 4D
        # At prediction time: (batch, pred_len, num_feat)
        # At training time: (batch, num_forking, pred_len, num_feat)
        if future_feat_dynamic.ndim == 3:
            # Prediction time: add forking dimension
            # Shape: (batch, pred_len, num_feat) -> (batch, 1, pred_len, num_feat)
            future_feat_dynamic = future_feat_dynamic.unsqueeze(1)
            # Repeat for num_forking positions (though we only use the last one)
            future_feat_dynamic = future_feat_dynamic.expand(
                -1, self.num_forking, -1, -1
            )

        # Flatten future features for decoder
        # Shape: (batch, num_forking, prediction_length * num_feat_dynamic_real)
        batch_size, num_forking, pred_len, num_feat = future_feat_dynamic.shape
        future_feat_flat = future_feat_dynamic.reshape(
            batch_size, num_forking, pred_len * num_feat
        )

        # Concatenate encoder output with future features
        # Shape: (batch, num_forking, encoder_output_size + pred_len * num_feat)
        dec_input_dynamic = torch.cat([enc_output_forking, future_feat_flat], dim=-1)

        # Decode
        # Shape: (batch, num_forking, prediction_length, final_dim)
        dec_output = self.decoder(enc_output_static, dec_input_dynamic)

        return dec_output, scale

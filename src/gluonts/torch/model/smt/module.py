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
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.scaler import Scaler, MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.util import (
    lagged_sequence_values,
    take_last,
    unsqueeze_expand,
)
from gluonts.model import Input, InputSpec


def make_transformer(
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.TransformerEncoder:
    """
    A pre-norm Transformer stack, used as the bidirectional encoder / memory
    cell and -- with a causal mask -- as the decoder. SMT's decoder reads the
    memory tokens as a prefix via causal self-attention, so this is a masked
    ``nn.TransformerEncoder``, not a cross-attention ``nn.TransformerDecoder``.
    """
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity regularizer on the memory tokens (Wang & Isola, 2020), as used
    by SMT to spread the memory representation over the hypersphere.

    ``z`` has shape ``(batch, num_tokens, dim)``; the loss is computed over the
    flattened set of (normalized) tokens.

    Implemented without ``torch.pdist`` (which lacks an MPS kernel): for unit
    vectors the squared pairwise distance is ``2 - 2 * <z_i, z_j>``.
    """
    z = F.normalize(z.reshape(-1, z.shape[-1]), dim=-1)
    sq_dist = (2.0 - 2.0 * (z @ z.t())).clamp(min=0.0)
    n = z.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=z.device)
    pairwise = sq_dist[iu[0], iu[1]]
    return pairwise.mul(-t).exp().mean().log()


class SMTModel(nn.Module):
    """
    Module implementing a forecasting model trained with Supervised Memory
    Training (SMT) [Kumar & Isola, 2026].

    Like DeepAR, the deployed model is a recurrent network with a distribution
    head: an initial memory state is produced from the context and rolled
    forward one step at a time, sampling autoregressively. Unlike DeepAR, the
    recurrent cell is *not* trained by backpropagation through time. Instead a
    Transformer *teacher* encodes a context window into a memory state ``m`` and
    is trained on a predictive-state objective (predict the future from ``m``);
    the recurrent cell is then trained by supervised regression onto the
    teacher's one-step memory transitions ``(m_t, x_{t+1}) -> m_{t+1}``.

    The feature pipeline (lags, time/age features, static embeddings and mean
    scaling) is identical to ``DeepARModel``; the time features double as the
    positional encoding, so no extra positional embedding is used.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: Optional[List[int]] = None,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_rnn_layers: int = 2,
        mem_tokens: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout_rate: float = 0.1,
        coef_dyn: float = 0.1,
        coef_unif: float = 0.001,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        default_scale: Optional[float] = None,
        num_parallel_samples: int = 100,
        nonnegative_pred_samples: bool = False,
    ) -> None:
        super().__init__()

        assert distr_output.event_shape == ()
        assert num_feat_dynamic_real > 0
        assert num_feat_static_real > 0
        assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert (
            embedding_dimension is None
            or len(embedding_dimension) == num_feat_static_cat
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.lags_seq = [l - 1 for l in self.lags_seq]
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler: Scaler = MeanScaler(
                dim=-1, keepdim=True, default_scale=default_scale
            )
        else:
            self.scaler = NOPScaler(dim=-1, keepdim=True)
        self.nonnegative_pred_samples = nonnegative_pred_samples

        self.d_model = d_model
        self.mem_tokens = mem_tokens
        self.coef_dyn = coef_dyn
        self.coef_unif = coef_unif
        dim_feedforward = dim_feedforward or 4 * d_model

        self.input_size = len(self.lags_seq) + self._number_of_features
        # embed each per-step feature vector into a token. The LayerNorm is
        # required, not optional: a near-constant context window floors the
        # mean scale to ~1e-10, which blows the scaled lag features up to ~1e10
        # -- an LSTM tolerates that, a Transformer does not.
        self.embed = nn.Linear(self.input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        # learned register tokens that read out the memory state from the
        # bidirectional encoder
        self.mem_register = nn.Parameter(
            torch.randn(mem_tokens, d_model) * 0.02
        )

        # teacher encoder (bidirectional): context features -> memory
        self.encoder = make_transformer(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout_rate
        )
        # shared causal decoder: predictive-state head during training and
        # single-step readout head at deployment
        self.decoder = make_transformer(
            d_model, nhead, num_decoder_layers, dim_feedforward, dropout_rate
        )
        # recurrent memory cell (bidirectional over [memory; input token])
        self.rnn_cell = make_transformer(
            d_model, nhead, num_rnn_layers, dim_feedforward, dropout_rate
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat),
                    dtype=torch.long,
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real),
                    dtype=torch.float,
                ),
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self._past_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self._past_length),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 1  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def prepare_rnn_input(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the per-step feature sequence (identical to ``DeepARModel``).

        Returns a tuple ``(features, scale, static_feat)`` where ``features``
        has shape ``(batch, context_length - 1 + future_length, input_size)``.
        """
        context = past_target[..., -self.context_length :]
        observed_context = past_observed_values[..., -self.context_length :]

        input, _, scale = self.scaler(context, observed_context)
        future_length = future_time_feat.shape[-2]
        if future_length > 1:
            assert future_target is not None
            input = torch.cat(
                (input, future_target[..., : future_length - 1] / scale),
                dim=-1,
            )
        prior_input = past_target[..., : -self.context_length] / scale

        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1
        )

        time_feat = torch.cat(
            (
                take_last(past_time_feat, dim=-2, num=self.context_length - 1),
                future_time_feat,
            ),
            dim=-2,
        )

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=-1,
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=time_feat.shape[-2]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        return torch.cat((lags, features), dim=-1), scale, static_feat

    def _causal_mask(self, size: int, device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(
            size, device=device
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode a window of per-step features into a memory state.

        ``features`` has shape ``(batch, length, input_size)``; returns the
        memory state of shape ``(batch, mem_tokens, d_model)``.
        """
        batch_size = features.shape[0]
        tokens = self.input_norm(self.embed(features))
        register = self.mem_register.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat((tokens, register), dim=1))
        memory = encoded[:, -self.mem_tokens :]
        return F.rms_norm(memory, (self.d_model,))

    def rnn_update(
        self, memory: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        One-step recurrent memory transition: incorporate the next per-step
        feature vector into the memory state.

        ``memory``: ``(batch, mem_tokens, d_model)``;
        ``features``: ``(batch, 1, input_size)``.
        """
        token = self.input_norm(self.embed(features))
        updated = self.rnn_cell(torch.cat((memory, token), dim=1))
        memory = updated[:, : self.mem_tokens]
        return F.rms_norm(memory, (self.d_model,))

    def decode(
        self, memory: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Predictive-state head: predict the distribution of the target at each
        future step from the memory state and the (teacher-forced) future
        features.

        ``memory``: ``(batch, mem_tokens, d_model)``;
        ``features``: ``(batch, future_length, input_size)``.

        The future feature at step ``j`` carries the realized target of step
        ``j - 1`` as a lag (teacher forcing); the causal output aligned with
        that feature predicts the target of step ``j``. Returns distribution
        arguments for ``future_length`` steps.
        """
        future_length = features.shape[1]
        tokens = self.input_norm(self.embed(features))
        sequence = torch.cat((memory, tokens), dim=1)
        mask = self._causal_mask(sequence.shape[1], sequence.device)
        decoded = self.decoder(sequence, mask=mask, is_causal=True)
        # the output aligned with future feature j predicts target j
        output = decoded[:, self.mem_tokens : self.mem_tokens + future_length]
        return self.param_proj(output)

    def post_process_samples(self, samples: torch.Tensor) -> torch.Tensor:
        if self.nonnegative_pred_samples:
            return torch.relu(samples)
        return samples

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> dict:
        features, scale, _ = self.prepare_rnn_input(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        # context positions [0, context_length - 1) summarize the past;
        # future positions are the prediction window.
        split = self.context_length - 1
        context_features = features[:, :split]
        future_features = features[:, split:]

        memory = self.encode(context_features)

        # predictive-state loss: NLL of the future targets under the teacher
        params = self.decode(memory, future_features)
        loss_pred = self.distr_output.loss(
            target=future_target, distr_args=params, scale=scale
        )
        loss_pred = (loss_pred * future_observed_values).sum() / (
            future_observed_values.sum().clamp(min=1.0)
        )

        # dynamics loss: the recurrent cell regresses onto the teacher's
        # one-step memory transition, with ``memory_next`` a detached label --
        # the label-matching gradient should not flow back into the teacher.
        memory_next = self.encode(features[:, 1 : split + 1])
        transition_feature = features[:, split : split + 1]
        memory_next_pred = self.rnn_update(memory, transition_feature)
        loss_dyn = F.mse_loss(memory_next_pred, memory_next.detach())

        loss_unif = uniformity_loss(memory)

        loss = (
            loss_pred + self.coef_dyn * loss_dyn + self.coef_unif * loss_unif
        )

        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_dyn": loss_dyn,
            "loss_unif": loss_unif,
        }

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Deploy the SMT recurrent network: roll the memory cell forward with
        O(1) state, sampling autoregressively.

        The encoder produces the initial memory ``m`` from the past (as in the
        official SMT, the teacher encoder seeds the RNN). Then, at each step,
        the next target is read out from the memory and the current step's
        feature, the sampled target is fed back as the next feature, and the
        memory is advanced by a single ``rnn_update`` (no growing history /
        no backprop-through-time) -- this is the network SMT actually trains.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        # encode the past (target + covariates) into the memory state
        features, scale, static_feat = self.prepare_rnn_input(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )
        split = self.context_length - 1
        memory = self.encode(features[:, :split])
        # the first feature carries the last context target as its lag and the
        # first future covariate.
        current_feature = features[:, split : split + 1]

        repeated_scale = scale.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_memory = memory.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        current_feature = current_feature.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        future_samples = []
        for k in range(self.prediction_length):
            # read out the next target from memory + current feature (O(1))
            params = self.decode(repeated_memory, current_feature)
            last_params = tuple(p[:, -1:] for p in params)
            distr = self.distr_output.distribution(
                last_params, scale=repeated_scale
            )
            next_sample = distr.sample()
            future_samples.append(next_sample)

            # advance the memory one step with the recurrent cell (O(1) state)
            repeated_memory = self.rnn_update(repeated_memory, current_feature)

            if k + 1 < self.prediction_length:
                scaled_next_sample = next_sample / repeated_scale
                next_lags = lagged_sequence_values(
                    self.lags_seq,
                    repeated_past_target,
                    scaled_next_sample,
                    dim=-1,
                )
                next_features = torch.cat(
                    (
                        repeated_static_feat,
                        repeated_time_feat[:, k + 1 : k + 2],
                    ),
                    dim=-1,
                )
                current_feature = torch.cat((next_lags, next_features), dim=-1)
                repeated_past_target = torch.cat(
                    (repeated_past_target, scaled_next_sample), dim=1
                )

        future_samples_concat = torch.cat(future_samples, dim=1)
        future_samples_concat = self.post_process_samples(
            future_samples_concat
        )

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
        )

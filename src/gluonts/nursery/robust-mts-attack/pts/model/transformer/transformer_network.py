from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import DistributionOutput
from pts.modules import MeanScaler, NOPScaler, FeatureEmbedder


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class TransformerNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_heads: int,
        act_type: str,
        dropout_rate: float,
        dim_feedforward_scale: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        cardinality: List[int],
        embedding_dimension: List[int],
        lags_seq: List[int],
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.scaling = scaling
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.distr_output = distr_output

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq

        self.target_shape = distr_output.event_shape

        # [B, T, input_size] -> [B, T, d_model]
        self.encoder_input = nn.Linear(input_size, d_model)
        self.decoder_input = nn.Linear(input_size, d_model)

        # [B, T, d_model] where d_model / num_heads is int
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_scale * d_model,
            dropout=dropout_rate,
            activation=act_type,
        )

        self.proj_dist_args = distr_output.get_args_proj(d_model)

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=embedding_dimension,
        )

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        # mask
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(prediction_length),
        )

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def create_network_input(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,
        # (batch_size, num_features, history_length)
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,  # (batch_size, history_length, 1)
        past_observed_values: torch.Tensor,  # (batch_size, history_length)
        future_time_feat: Optional[
            torch.Tensor
        ],  # (batch_size, num_features, prediction_length)
        # (batch_size, prediction_length)
        future_target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates inputs for the transformer network.
        All tensor arguments should have NTC layout.
        """

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat[
                :, self.history_length - self.context_length :, ...
            ]
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, self.history_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            sequence = torch.cat((past_target, future_target), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                torch.log(scale)
                if len(self.target_shape) == 0
                else torch.log(scale.squeeze(1)),
            ),
            dim=1,
        )

        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags_scaled.reshape(
            (-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape))
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = torch.cat((input_lags, time_feat, repeated_static_feat), dim=-1)

        return inputs, scale, static_feat


class TransformerTrainingNetwork(TransformerNetwork):
    # noinspection PyMethodOverriding,PyPep8Naming
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss for training Transformer, all inputs tensors representing time series have NTC layout.
        Parameters
        ----------
        feat_static_cat : (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape, seq_len)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, *target_shape)
        Returns
        -------
        Loss with shape (batch_size, context + prediction_length, 1)
        """

        # create the inputs for the encoder
        inputs, scale, _ = self.create_network_input(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        enc_input = inputs[:, : self.context_length, ...]  # F.slice_axis(
        #     inputs, axis=1, begin=0, end=self.context_length
        # )
        dec_input = inputs[:, self.context_length :, ...]  # F.slice_axis(
        #     inputs, axis=1, begin=self.context_length, end=None
        # )

        # pass through encoder [T, B, b_model]
        enc_out = self.transformer.encoder(
            self.encoder_input(enc_input).permute(1, 0, 2)
        )

        # input to decoder
        dec_output = self.transformer.decoder(
            self.decoder_input(dec_input).permute(1, 0, 2),
            enc_out,  # memory
            tgt_mask=self.tgt_mask,
        )

        # compute loss
        distr_args = self.proj_dist_args(dec_output.permute(1, 0, 2))
        distr = self.distr_output.distribution(distr_args, scale=scale)
        loss = -distr.log_prob(future_target)

        return loss.mean()


class TransformerPredictionNetwork(TransformerNetwork):
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        static_feat: torch.Tensor,
        past_target: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        enc_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes sample paths by decoding from the transformer.
        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length, 1).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, ).
        enc_out: Tensor
            output of the encoder. Shape: (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            a tensor containing sampled paths. Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_time_feat = time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(1)

        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=1
        )

        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = lags / repeated_scale.unsqueeze(1)
            # lags_scaled = F.broadcast_div(
            #     lags, repeated_scale.expand_dims(axis=-1)
            # )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = lags_scaled.reshape(
                shape=(-1, 1, prod(self.target_shape) * len(self.lags_seq))
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            dec_input = torch.cat(
                (input_lags, repeated_time_feat[:, k : k + 1, :], repeated_static_feat),
                dim=-1,
            )

            dec_output = self.transformer.decoder(
                self.decoder_input(dec_input).permute(1, 0, 2), repeated_enc_out
            )

            distr_args = self.proj_dist_args(dec_output.permute(1, 0, 2))

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample()

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = torch.cat((repeated_past_target, new_samples), dim=1)
            future_samples.append(new_samples)

        # reset cache of the decoder
        # self.transformer.decoder.cache_reset()

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, *target_shape, prediction_length)
        return samples.reshape(
            (
                (-1, self.num_parallel_samples)
                + self.target_shape
                + (self.prediction_length,)
            )
        )

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)
        Returns predicted samples
        -------
        """

        # create the inputs for the encoder
        inputs, scale, static_feat = self.create_network_input(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=None,
            future_target=None,
        )

        # pass through encoder
        enc_out = self.transformer.encoder(self.encoder_input(inputs).permute(1, 0, 2))

        return self.sampling_decoder(
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            enc_out=enc_out,
        )

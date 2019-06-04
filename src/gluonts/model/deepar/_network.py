# Standard library imports
from typing import List, Optional, Tuple

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.block.feature import FeatureEmbedder
from gluonts.block.scaler import MeanScaler, NOPScaler
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput
from gluonts.model.common import Tensor
from gluonts.support.util import weighted_average


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class DeepARNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: int,
        lags_seq: List[int],
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling

        assert len(set(lags_seq)) == len(
            lags_seq
        ), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq

        self.distr_output = distr_output
        RnnCell = {'lstm': mx.gluon.rnn.LSTMCell, 'gru': mx.gluon.rnn.GRUCell}[
            self.cell_type
        ]

        self.target_shape = distr_output.event_shape

        # TODO: is the following restriction needed?
        assert (
            len(self.target_shape) <= 1
        ), "Argument `target_shape` should be a tuple with 1 element at most"

        with self.name_scope():
            self.proj_distr_args = distr_output.get_args_proj()
            self.rnn = mx.gluon.rnn.HybridSequentialRNNCell()
            for k in range(num_layers):
                cell = RnnCell(hidden_size=num_cells)
                cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
                cell = (
                    mx.gluon.rnn.ZoneoutCell(cell, zoneout_states=dropout_rate)
                    if dropout_rate > 0.0
                    else cell
                )
                self.rnn.add(cell)
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension for _ in cardinality],
            )
            if scaling:
                self.scaler = MeanScaler(keepdims=True)
            else:
                self.scaler = NOPScaler(keepdims=True)

    @staticmethod
    def get_lagged_subsequences(
        F,
        sequence: Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted. Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and I = len(indices), containing lagged
            subsequences. Specifically, lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: sequence_length - lag_index - subsequences_length >= 0
        # for all lag_index, hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                F.slice_axis(
                    sequence, axis=1, begin=begin_index, end=end_index
                )
            )

        return F.stack(*lagged_values, axis=-1)

    @staticmethod
    def weighted_average(
        F, tensor: Tensor, weights: Optional[Tensor] = None, axis=None
    ):
        if weights is not None:
            weighted_tensor = tensor * weights
            sum_weights = F.maximum(1.0, weights.sum(axis=axis))
            return weighted_tensor.sum(axis=axis) / sum_weights
        else:
            return tensor.mean(axis=axis)

    def unroll_encoder(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            Tensor
        ],  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Unrolls the LSTM encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of past_target
        and a vector of static features that was constructed and fed as input
        to the encoder.
        All tensor arguments should have NTC layout.
        """

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=None,
            )
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=None,
                ),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target, future_target, dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags = self.get_lagged_subsequences(
            F=F,
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = F.concat(
            embedded_cat,
            F.log(scale)
            if len(self.target_shape) == 0
            else F.log(scale.squeeze(axis=1)),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.expand_dims(axis=1).repeat(
            axis=1, repeats=subsequences_length
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = F.reshape(
            data=lags_scaled,
            shape=(
                -1,
                subsequences_length,
                len(self.lags_seq) * prod(self.target_shape),
            ),
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = F.concat(input_lags, time_feat, repeated_static_feat, dim=-1)

        # unroll encoder
        outputs, state = self.rnn.unroll(
            inputs=inputs,
            length=subsequences_length,
            layout="NTC",
            merge_outputs=True,
        )

        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))
        return outputs, state, scale, static_feat


class DeepARTrainingNetwork(DeepARNetwork):

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Computes the loss for training DeepAR, all inputs tensors representing time series have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape, seq_len)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, *target_shape)
        future_observed_values : (batch_size, prediction_length, *target_shape)

        Returns loss with shape (batch_size, context + prediction_length, 1)
        -------

        """

        # unroll the decoder in "training mode", i.e. by providing future data as well
        rnn_outputs, _, scale, _ = self.unroll_encoder(
            F=F,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        # put together target sequence
        # (batch_size, seq_len, *target_shape)
        target = F.concat(
            past_target.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=None,
            ),
            future_target,
            dim=1,
        )

        distr_args = self.proj_distr_args(rnn_outputs)

        # compute distribution
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # (batch_size, seq_len)
        loss = distr.loss(target)

        # (batch_size, seq_len, *target_shape)
        observed_values = F.concat(
            past_observed_values.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=self.history_length,
            ),
            future_observed_values,
            dim=1,
        )

        # mask the loss at one time step iff one or more observations is missing in the target dimensions
        # (batch_size, seq_len)
        loss_weights = (
            observed_values
            if (len(self.target_shape) == 0)
            else observed_values.min(axis=-1, keepdims=False)
        )

        weighted_loss = weighted_average(
            F=F, x=loss, weights=loss_weights, axis=1
        )

        return (weighted_loss, loss) + distr_args


class DeepARPredictionNetwork(DeepARNetwork):
    @validated()
    def __init__(self, num_sample_paths: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_sample_paths = num_sample_paths

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        F,
        static_feat: Tensor,
        past_target: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        begin_states: List,
    ) -> Tensor:
        """
        Computes sample paths by unrolling the LSTM starting with a initial input and state.
        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, 1, 1).
        begin_states : List
            list of initial states for the LSTM layers.
            the shape of each tensor of the list should be (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            a tensor containing sampled paths. Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target = past_target.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_time_feat = time_feat.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_static_feat = static_feat.repeat(
            repeats=self.num_sample_paths, axis=0
        ).expand_dims(axis=1)
        repeated_scale = scale.repeat(repeats=self.num_sample_paths, axis=0)
        repeated_states = [
            s.repeat(repeats=self.num_sample_paths, axis=0)
            for s in begin_states
        ]

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                F=F,
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = F.broadcast_div(
                lags, repeated_scale.expand_dims(axis=-1)
            )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = F.reshape(
                data=lags_scaled,
                shape=(-1, 1, prod(self.target_shape) * len(self.lags_seq)),
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input = F.concat(
                input_lags,
                repeated_time_feat.slice_axis(axis=1, begin=k, end=k + 1),
                repeated_static_feat,
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, repeated_states = self.rnn.unroll(
                inputs=decoder_input,
                length=1,
                begin_state=repeated_states,
                layout="NTC",
                merge_outputs=True,
            )

            distr_args = self.proj_distr_args(rnn_outputs)

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(
                distr_args, scale=repeated_scale
            )

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample()

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = F.concat(
                repeated_past_target, new_samples, dim=1
            )
            future_samples.append(new_samples)

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = F.concat(*future_samples, dim=1)

        # (batch_size, num_samples, *target_shape, prediction_length)
        return samples.reshape(
            shape=(
                (-1, self.num_sample_paths)
                + self.target_shape
                + (self.prediction_length,)
            )
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Tensor,  # (batch_size, prediction_length, num_features)
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns predicted samples
        -------

        """

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, state, scale, static_feat = self.unroll_encoder(
            F=F,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=None,
            future_target=None,
        )

        return self.sampling_decoder(
            F=F,
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            begin_states=state,
        )

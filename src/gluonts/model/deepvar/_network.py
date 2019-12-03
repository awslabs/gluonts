# Standard library imports
from typing import List, Optional, Tuple

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.block.scaler import NOPScaler, MeanScaler
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput
from gluonts.model.common import Tensor
from gluonts.support.util import weighted_average, assert_shape


def make_rnn_cell(
    num_cells: int,
    num_layers: int,
    cell_type: str,
    residual: bool,
    dropout_rate: float,
):
    RnnCell = {"lstm": mx.gluon.rnn.LSTMCell, "gru": mx.gluon.rnn.GRUCell}[
        cell_type
    ]
    rnn = mx.gluon.rnn.HybridSequentialRNNCell()
    for k in range(num_layers):
        cell = RnnCell(hidden_size=num_cells)
        if residual:
            cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
        cell = (
            mx.gluon.rnn.ZoneoutCell(cell, zoneout_states=dropout_rate)
            if dropout_rate > 0.0
            else cell
        )
        rnn.add(cell)
    return rnn


class DeepVARNetwork(mx.gluon.HybridBlock):
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
        lags_seq: List[int],
        target_dim: int,
        conditioning_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
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
        self.target_dim = target_dim
        self.scaling = scaling
        self.target_dim_sample = target_dim
        self.conditioning_length = conditioning_length

        assert len(set(lags_seq)) == len(
            lags_seq
        ), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq

        self.distr_output = distr_output

        self.target_dim = target_dim

        with self.name_scope():
            self.proj_dist_args = distr_output.get_args_proj()

            residual = True

            self.rnn = make_rnn_cell(
                cell_type=cell_type,
                num_cells=num_cells,
                num_layers=num_layers,
                residual=residual,
                dropout_rate=dropout_rate,
            )

            self.embed_dim = 1
            self.embed = mx.gluon.nn.Embedding(
                input_dim=self.target_dim, output_dim=self.embed_dim
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
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
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
                ).expand_dims(axis=1)
            )
        return F.concat(
            *lagged_values, num_args=len(indices), dim=1
        ).transpose(axes=(0, 2, 3, 1))

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

    def unroll(
        self,
        F,
        lags: Tensor,
        scale: Tensor,
        time_feat: Tensor,
        target_dimensions: Tensor,
        unroll_length: int,
        begin_state: Optional[List[Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        Parameters
        ----------
        F
        lags : (batch_size, sub_seq_len, target_dim, num_lags)
        scale : (batch_size, 1, target_dim)
        time_feat :
        target_dimensions : (batch_size, target_dim)
        unroll_length : length to unroll

        Returns
        -------
        outputs : (batch_size, seq_len, num_cells)
        states : list of list of (batch_size, num_cells) tensors with
        dimensions: target_dim x num_layers x (batch_size, num_cells)
        lags_scaled : (batch_size, sub_seq_len, target_dim, num_lags)
        """
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        assert_shape(
            lags_scaled,
            (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        )

        input_lags = F.reshape(
            data=lags_scaled,
            shape=(-1, unroll_length, len(self.lags_seq) * self.target_dim),
        )

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimensions)
        assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

        # (batch_size, seq_len, target_dim * embed_dim)
        repeated_index_embeddings = (
            index_embeddings.expand_dims(axis=1)
            .repeat(axis=1, repeats=unroll_length)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = F.concat(
            input_lags, repeated_index_embeddings, time_feat, dim=-1
        )

        # unroll encoder
        outputs, state = self.rnn.unroll(
            inputs=inputs,
            length=unroll_length,
            layout="NTC",
            merge_outputs=True,
            begin_state=begin_state,
        )

        assert_shape(outputs, (-1, unroll_length, self.num_cells))
        for s in state:
            assert_shape(s, (-1, self.num_cells))

        assert_shape(
            lags_scaled,
            (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        )

        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        F,
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target_cdf: Tensor,  # (batch_size, history_length, target_dim)
        past_observed_values: Tensor,  # (batch_size, history_length, target_dim)
        past_is_pad: Tensor,  # (batch_size, history_length)
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, prediction_length, num_features)
        future_target_cdf: Optional[
            Tensor
        ],  # (batch_size, prediction_length, target_dim)
        target_dimensions: Tensor,
        is_train: bool,
    ) -> Tuple[Tensor, List[Tensor], Tensor, Tensor, Tensor]:
        """
        Unrolls the LSTM encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of past_target_cdf
        and a vector of static features that was constructed and fed as input
        to the encoder.
        All tensor arguments should have NTC layout.
        """

        past_observed_values = F.broadcast_minimum(
            past_observed_values, 1 - past_is_pad.expand_dims(axis=-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat.slice_axis(
                axis=1, begin=-self.context_length, end=None
            )
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(
                    axis=1, begin=-self.context_length, end=None
                ),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target_cdf, future_target_cdf, dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = self.get_lagged_subsequences(
            F=F,
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )

        outputs, states, lags_scaled, inputs = self.unroll(
            F=F,
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            target_dimensions=target_dimensions,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        # outputs: (batch_size, seq_len, target_dim)
        # states: list of list of (batch_size, num_cells) tensors with dimensions: target_dim x num_layers x (batch_size, num_cells)
        # scale: (batch_size, 1, target_dim)
        # lags_scaled: (batch_size, target_dim, sub_seq_len, num_lags)

        return outputs, states, scale, lags_scaled, inputs

    def distr(
        self,
        rnn_outputs: Tensor,
        scale: Tensor,
    ):
        """

        Parameters
        ----------
        rnn_outputs : (batch_size, seq_len, num_cells)
        time_features : (batch_size, seq_len, num_features)
        scale : (batch_size, 1, target_dim)
        lags_scaled : (batch_size, seq_len, target_dim, num_lags)
        target_dimensions : (batch_size, target_dim)
        seq_len: length of the sequences

        Returns
        -------

        """
        distr_args = self.proj_dist_args(rnn_outputs)

        # compute likelihood of target given the predicted parameters
        distr = self.distr_output.distribution(distr_args, scale=scale)

        return distr, distr_args

    def train_hybrid_forward(
        self,
        F,
        target_dimensions: Tensor,
        past_time_feat: Tensor,
        past_target_cdf: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target_cdf: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:

        seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data as well
        rnn_outputs, _, scale, lags_scaled, inputs = self.unroll_encoder(
            F=F,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimensions=target_dimensions,
            is_train=True,
        )

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        target = F.concat(
            past_target_cdf.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            future_target_cdf,
            dim=1,
        )

        # assert_shape(target, (-1, seq_len, self.target_dim))

        distr, distr_args = self.distr(
            rnn_outputs=rnn_outputs,
            scale=scale,
        )

        # we sum the last axis to have the same shape for all likelihoods
        # (batch_size, subseq_length, 1)
        likelihoods = -distr.log_prob(target).expand_dims(axis=-1)

        assert_shape(likelihoods, (-1, seq_len, 1))

        past_observed_values = F.broadcast_minimum(
            past_observed_values, 1 - past_is_pad.expand_dims(axis=-1)
        )

        # (batch_size, subseq_length, target_dim)
        observed_values = F.concat(
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            future_observed_values,
            dim=1,
        )

        # mask the loss at one time step if one or more observations is missing in the target dimensions
        # (batch_size, subseq_length, 1)
        loss_weights = observed_values.min(axis=-1, keepdims=True)

        assert_shape(loss_weights, (-1, seq_len, 1))

        loss = weighted_average(
            F=F, x=likelihoods, weights=loss_weights, axis=1
        )

        assert_shape(loss, (-1, -1, 1))

        self.distribution = distr

        return (loss, likelihoods) + distr_args


class DeepVARTrainingNetwork(DeepVARNetwork):

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        target_dimensions: Tensor,
        past_time_feat: Tensor,
        past_target_cdf: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target_cdf: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing time series have NTC layout.

        Parameters
        ----------
        F
        target_dimensions: indices of the target dimension (batch_size, num_observations)
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target_cdf : (batch_size, history_length, target_dim)
        past_observed_values : (batch_size, history_length, target_dim, seq_len)
        past_is_pad : (batch_size, history_length)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target_cdf : (batch_size, prediction_length, target_dim)
        future_observed_values : (batch_size, prediction_length, target_dim)

        Returns loss with shape (batch_size, context + prediction_length, 1)
        -------

        """
        return self.train_hybrid_forward(
            F,
            target_dimensions,
            past_time_feat,
            past_target_cdf,
            past_observed_values,
            past_is_pad,
            future_time_feat,
            future_target_cdf,
            future_observed_values,
        )


class DeepVARPredictionNetwork(DeepVARNetwork):
    @validated()
    def __init__(self, num_sample_paths: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_sample_paths = num_sample_paths

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]



    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        target_dimensions: Tensor,
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target_cdf: Tensor,  # (batch_size, history_length, target_dim)
        past_observed_values: Tensor,  # (batch_size, history_length, target_dim)
        past_is_pad: Tensor,
        future_time_feat: Tensor,  # (batch_size, prediction_length, num_features)
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        F
        past_time_feat : (batch_size, history_length, num_features)
        past_target_cdf : (batch_size, history_length, target_dim)
        target_dimensions : (batch_size, target_dim)
        past_observed_values : (batch_size, history_length, target_dim)
        past_is_pad : (batch_size, history_length)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns predicted samples
        -------

        """
        return self.predict_hybrid_forward(
            F=F,
            target_dimensions=target_dimensions,
            past_time_feat=past_time_feat,  # (batch_size, history_length, num_features)
            past_target_cdf=past_target_cdf,  # (batch_size, history_length, target_dim)
            past_observed_values=past_observed_values,  # (batch_size, history_length, target_dim)
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,  # (batch_size, prediction_length, num_features)
        )

    def sampling_decoder(
        self,
        F,
        past_target_cdf: Tensor,
        target_dimensions: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        begin_states: List[Tensor],
    ) -> Tensor:
        """
        Computes sample paths by unrolling the LSTM starting with a initial input and state.
        Parameters
        ----------
        past_target_cdf : Tensor
            target history. Shape: (batch_size, history_length).
        target_dimensions : (batch_size, target_dim)
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

        def repeat(tensor):
            return tensor.repeat(repeats=self.num_sample_paths, axis=0)

        # blows-up the dimension of each tensor to batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        repeated_target_dimensions = repeat(target_dimensions)

        # slight difference for GPVAR and DeepVAR, in GPVAR, its a list
        repeated_states = self.make_states(begin_states)

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                F=F,
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, lags_scaled, inputs = self.unroll(
                F=F,
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat.slice_axis(
                    axis=1, begin=k, end=k + 1
                ),
                target_dimensions=repeated_target_dimensions,
                unroll_length=1,
            )

            distr, distr_args = self.distr(
                rnn_outputs=rnn_outputs,
                scale=repeated_scale,
            )

            # (batch_size, 1, target_dim)
            new_samples = distr.sample()

            # (batch_size, seq_len, target_dim)
            future_samples.append(new_samples)
            repeated_past_target_cdf = F.concat(
                repeated_past_target_cdf, new_samples, dim=1
            )

        # (batch_size * num_samples, prediction_length, target_dim)
        samples = F.concat(*future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            shape=(
                -1,
                self.num_sample_paths,
                self.prediction_length,
                self.target_dim,
            )
        )

    def make_states(self, begin_states):
        def repeat(tensor):
            return tensor.repeat(repeats=self.num_sample_paths, axis=0)

        return [repeat(s) for s in begin_states]

    def predict_hybrid_forward(
        self,
        F,
        target_dimensions: Tensor,
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target_cdf: Tensor,  # (batch_size, history_length, target_dim)
        past_observed_values: Tensor,  # (batch_size, history_length, target_dim)
        past_is_pad: Tensor,
        future_time_feat: Tensor,  # (batch_size, prediction_length, num_features)
    ) -> Tensor:

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = F.broadcast_minimum(
            past_observed_values, 1 - past_is_pad.expand_dims(axis=-1)
        )

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, state, scale, _, inputs = self.unroll_encoder(
            F=F,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimensions=target_dimensions,
            is_train=False,
        )

        return self.sampling_decoder(
            F=F,
            past_target_cdf=past_target_cdf,
            target_dimensions=target_dimensions,
            time_feat=future_time_feat,
            scale=scale,
            begin_states=state,
        )

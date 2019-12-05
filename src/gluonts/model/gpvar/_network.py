# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.distribution.distribution import getF
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.model.deepvar._network import DeepVARNetwork


class GPVARNetwork(DeepVARNetwork):
    @validated()
    def __init__(self, target_dim_sample: int, **kwargs) -> None:
        super().__init__(embedding_dimension=1, cardinality=[1], **kwargs)
        self.target_dim_sample = target_dim_sample

        with self.name_scope():
            self.embed = mx.gluon.nn.Embedding(
                input_dim=self.target_dim,
                output_dim=4 * self.distr_output.rank,
            )

    def unroll(
        self,
        F,
        lags,
        scale,
        time_feat,
        begin_state,
        target_dimensions,
        unroll_length,
    ):
        """

        Parameters
        ----------
        F
        lags : (batch_size, sub_seq_len, target_dim, num_lags)
        scale : (batch_size, 1, target_dim)
        time_feat : (batch_size, sub_seq_len, num_features)
        begin_state : list of state for each dimension
        target_dimensions : (batch_size, target_dim)
        unroll_length : length to unroll

        Returns
        -------
        outputs : (batch_size, seq_len, target_dim, num_cells)
        states : list of list of (batch_size, num_cells) tensors with dimensions: target_dim x num_layers x (batch_size, num_cells)
        lags_scaled : (batch_size, sub_seq_len, target_dim, num_lags)
        """
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        outputs = []
        states = []

        # from (batch_size, sub_seq_len, target_dim, num_lags) to # (batch_size, seq_len, target_dim * num_lags)
        # all_input_lags = F.reshape(
        #    data=lags_scaled,
        #    shape=(-1, unroll_length, self.target_dim * len(self.lags_seq)),
        # )

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimensions)

        # (batch_size, seq_len, target_dim, embed_dim)
        repeated_index_embeddings = index_embeddings.expand_dims(
            axis=1
        ).repeat(axis=1, repeats=unroll_length)

        inputs_seq = []

        for i in range(self.target_dim_sample):
            # (batch_size, sub_seq_len, input_dim)
            inputs = F.concat(
                lags_scaled.slice_axis(axis=2, begin=i, end=i + 1).squeeze(
                    axis=2
                ),
                # lags_std.slice_axis(axis=2, begin=i, end=i + 1).squeeze(axis=2),
                repeated_index_embeddings.slice_axis(
                    axis=2, begin=i, end=i + 1
                ).squeeze(axis=2),
                # all_input_lags,
                time_feat,
                dim=-1,
            )

            # unroll encoder
            # outputs: (batch_size, sub_seq_len, num_hidden)
            # state: tuple of (batch_size, num_hidden)
            outputs_single_dim, state = self.rnn.unroll(
                inputs=inputs,
                length=unroll_length,
                layout="NTC",
                merge_outputs=True,
                begin_state=begin_state[i]
                if begin_state is not None
                else None,
            )
            outputs.append(outputs_single_dim)
            states.append(state)
            inputs_seq.append(inputs)

        # (batch_size, seq_len, target_dim, num_cells)
        outputs = F.stack(*outputs, num_args=self.target_dim_sample, axis=2)

        return outputs, states, lags_scaled, time_feat

    def distr(
        self,
        rnn_outputs: Tensor,
        time_features: Tensor,
        scale: Tensor,
        lags_scaled: Tensor,
        target_dimensions: Tensor,
        seq_len: int,
    ):
        """

        Parameters
        ----------
        rnn_outputs : (batch_size, seq_len, num_cells)
        time_features : (batch_size, seq_len, num_features)
        scale : (batch_size, 1, target_dim)
        lags_scaled : (batch_size, seq_len, target_dim, num_lags)
        target_dimensions : (batch_size, target_dim)

        Returns
        -------

        """
        F = getF(rnn_outputs)

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimensions)

        # broadcast to (batch_size, seq_len, target_dim, embed_dim)
        repeated_index_embeddings = index_embeddings.expand_dims(
            axis=1
        ).repeat(axis=1, repeats=seq_len)

        # broadcast to (batch_size, seq_len, target_dim, num_features)
        time_features = time_features.expand_dims(axis=2).repeat(
            axis=2, repeats=self.target_dim_sample
        )

        # (batch_size, seq_len, target_dim, embed_dim + num_cells + num_inputs)
        distr_input = F.concat(
            rnn_outputs, repeated_index_embeddings, time_features, dim=-1
        )

        # TODO 1 pass inputs in proj args
        distr_args = self.proj_dist_args(distr_input)

        # compute likelihood of target given the predicted parameters
        distr = self.distr_output.distribution(
            distr_args, scale=scale, dim=self.target_dim_sample
        )

        return distr, distr_args


class GPVARTrainingNetwork(GPVARNetwork):

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
        Computes the loss for training GPVAR, all inputs tensors representing time series have NTC layout.

        Parameters
        ----------
        F
        target_dimensions: indices of the target dimension (batch_size, num_observations)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, target_dim)
        past_observed_values : (batch_size, history_length, target_dim, seq_len)
        past_is_pad : (batch_size, history_length)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, target_dim)
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


class GPVARPredictionNetwork(GPVARNetwork):
    @validated()
    def __init__(self, num_sample_paths: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_sample_paths = num_sample_paths

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def make_states(self, begin_states):
        def repeat(tensor):
            return tensor.repeat(repeats=self.num_sample_paths, axis=0)

        return [[repeat(s) for s in states] for states in begin_states]

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
        past_target : (batch_size, history_length, target_dim)
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

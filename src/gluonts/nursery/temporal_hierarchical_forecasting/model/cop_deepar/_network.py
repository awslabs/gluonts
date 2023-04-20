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

from typing import Dict, List, Optional, Tuple

import mxnet as mx
import numpy as np

from gluonts.core.component import Type, validated
from gluonts.itertools import prod
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.deepar._network import DeepARPredictionNetwork
from gluonts.mx.model.deepvar_hierarchical._estimator import (
    constraint_mat,
    null_space_projection_mat,
)
from gluonts.mx.model.deepvar_hierarchical._network import coherency_error
from gluonts.mx.distribution import Distribution, EmpiricalDistribution
from gluonts.mx import Tensor
from gluonts.mx.distribution import TransformedPiecewiseLinear

from gluonts.nursery.temporal_hierarchical_forecasting.utils import utils
from gluonts.nursery.temporal_hierarchical_forecasting.model.cop_deepar import (
    gluonts_fixes,
    gnn,
)


def reconcile_samples(
    reconciliation_mat: Tensor,
    samples: Tensor,
    non_negative: bool = False,
    num_iters: int = 10,
) -> Tensor:
    if not non_negative:
        return mx.nd.dot(samples, reconciliation_mat, transpose_b=True)
    else:
        # Dykstra's projection method: Projection onto the intersection of convex sets.
        x = samples
        p = mx.nd.zeros_like(x)
        q = mx.nd.zeros_like(x)
        for _ in range(num_iters):
            # Projection onto the non-negative orthant.
            y = mx.nd.relu(x + p)
            p = x + p - y

            # Projection onto the null space.
            x = mx.nd.dot(y + q, reconciliation_mat, transpose_b=True)
            q = y + q - x

        return x


class COPNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        estimators: List[DeepAREstimator],
        prediction_length: int,
        temporal_hierarchy: utils.TemporalHierarchy,
        do_reconciliation: bool,
        dtype: Type,
        use_gnn: bool,
        use_mlp: bool,
        adj_mat_option: str,
        non_negative: bool = False,
        naive_reconciliation: bool = False,
        prediction: bool = False,
        loss_function: str = "crps_univariate",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.prediction_length = prediction_length
        self.temporal_hierarchy = temporal_hierarchy
        self.use_gnn = use_gnn
        self.use_mlp = use_mlp
        self.adj_mat_option = adj_mat_option
        self.do_reconciliation = do_reconciliation
        self.non_negative = non_negative
        self.loss_function = loss_function
        self.dtype = dtype

        A = constraint_mat(self.temporal_hierarchy.agg_mat)
        if naive_reconciliation:
            M = utils.naive_reconcilation_mat(
                self.temporal_hierarchy.agg_mat, self.temporal_hierarchy.nodes
            )
        else:
            M = null_space_projection_mat(A)
        self.M, self.A = mx.nd.array(M), mx.nd.array(A)

        self.estimators = estimators

        self.models = []
        with self.name_scope():
            for estimator in estimators:
                if not prediction:
                    self.network = estimator.create_training_network()
                else:
                    self.network = gluonts_fixes.create_prediction_network(
                        estimator
                    )

                self.register_child(self.network)
                self.models.append(self.network)

            if self.use_gnn:
                # GNN Layer: Do message passing for `L-1` times, where `L` is the number of levels of the hierarchy.
                self.gnn = gnn.GNN(
                    units=self.estimators[0].num_cells,
                    num_layers=len(self.temporal_hierarchy.agg_multiples) - 1,
                    adj_matrix=mx.nd.array(
                        self.temporal_hierarchy.adj_mat(
                            option=self.adj_mat_option
                        )
                    ),
                    use_mlp=self.use_mlp,
                )

    def get_target_related_feat_at_agg_level(
        self,
        agg_level: int,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_target: Optional[Tensor] = None,
        future_observed_values: Optional[Tensor] = None,
    ) -> Dict:
        """
        Aggregate target at the given aggregate level along with updating observed value and pad indicators.

        :param agg_level:
        :param past_target:
        :param past_observed_values:
        :param past_is_pad:
        :param future_target:
        :param future_observed_values:
        :return:
        """
        agg_multiple = self.temporal_hierarchy.agg_multiples[agg_level]

        # Truncating the history length of the base time series to the nearest multiple.
        base_history_length = (
            past_target.shape[1] // agg_multiple
        ) * agg_multiple

        past_target_agg = (
            utils.agg_series(
                past_target.slice_axis(
                    axis=1, begin=-base_history_length, end=None
                ),
                agg_multiple=agg_multiple,
            )
            .squeeze(axis=-1)
            .slice_axis(
                axis=1, begin=-self.models[agg_level].history_length, end=None
            )
        )

        past_is_pad_agg = (
            utils.agg_series(
                past_is_pad.slice_axis(
                    axis=1, begin=-base_history_length, end=None
                ),
                agg_multiple=agg_multiple,
            )
            .squeeze(axis=-1)
            .slice_axis(
                axis=1, begin=-self.models[agg_level].history_length, end=None
            )
        )

        past_is_pad_agg = mx.nd.where(
            past_is_pad_agg == 0.0,
            mx.nd.zeros_like(past_is_pad_agg),
            mx.nd.ones_like(past_is_pad_agg),
        )

        past_observed_values_agg = (
            utils.agg_series(
                past_observed_values.slice_axis(
                    axis=1, begin=-base_history_length, end=None
                ),
                agg_multiple=agg_multiple,
            )
            .squeeze(axis=-1)
            .slice_axis(
                axis=1, begin=-self.models[agg_level].history_length, end=None
            )
        )

        past_observed_values_agg = mx.nd.where(
            # We sum observed values of base time series at `agg_multiple` time steps;
            # if all of them are 1, then the observed value for the aggregated time series is 1 and 0 otherwise.
            # We could redefine agg_series to actually compute mean, but overloading that term might cause other
            # problems later.
            past_observed_values_agg == agg_multiple,
            mx.nd.ones_like(past_observed_values_agg),
            mx.nd.zeros_like(past_observed_values_agg),
        )

        target_related_feat_agg = {
            "past_target": past_target_agg,
            "past_is_pad": past_is_pad_agg,
            "past_observed_values": past_observed_values_agg,
        }
        if future_target is not None:
            future_target_agg = utils.agg_series(
                future_target, agg_multiple=agg_multiple
            ).squeeze(axis=-1)

            future_observed_values_agg = utils.agg_series(
                future_observed_values, agg_multiple=agg_multiple
            ).squeeze(axis=-1)

            future_observed_values_agg = mx.nd.where(
                future_observed_values_agg == agg_multiple,
                mx.nd.ones_like(future_observed_values_agg),
                mx.nd.zeros_like(future_observed_values_agg),
            )

            target_related_feat_agg.update(
                {
                    "future_target": future_target_agg,
                    "future_observed_values": future_observed_values_agg,
                }
            )

        return target_related_feat_agg

    def _embeddings_to_distr(
        self,
        F,
        embeddings_at_all_levels: Tensor,
        scales: List,
    ) -> Distribution:
        distr_output = self.models[0].distr_output
        distr_args_at_all_levels: Dict = {
            arg_name: [] for arg_name in distr_output.args_dim.keys()
        }
        scales_ls = []

        start_ix = 0
        for i, num_nodes in enumerate(
            self.temporal_hierarchy.num_nodes_per_level
        ):
            end_ix = start_ix + num_nodes
            distr_args = self.models[i].proj_distr_args(
                embeddings_at_all_levels[..., start_ix:end_ix, :]
            )

            for j, arg_ls in enumerate(distr_args_at_all_levels.values()):
                arg_ls.append(distr_args[j])

            scales_ls.append(scales[i].broadcast_like(distr_args[0]))

            start_ix = end_ix

        # Last dimension contains parameters at all time-levels and aggregation can be done on it.
        distr_args_at_all_levels = {
            arg_name: F.concat(*arg_ls, dim=-1)
            for arg_name, arg_ls in distr_args_at_all_levels.items()
        }

        scale_at_all_levels = F.concat(*scales_ls, dim=-1)

        distr_at_all_levels = distr_output.distribution(
            distr_args=distr_args_at_all_levels.values(),
            scale=scale_at_all_levels,
        )

        if isinstance(distr_at_all_levels, TransformedPiecewiseLinear):
            distr_at_all_levels = TransformedPiecewiseLinear(
                base_distribution=gluonts_fixes.PiecewiseLinearWithSampling(
                    gamma=distr_at_all_levels.base_distribution.gamma,
                    slopes=distr_at_all_levels.base_distribution.slopes,
                    knot_spacings=distr_at_all_levels.base_distribution.knot_spacings,
                ),
                transforms=distr_at_all_levels.transforms,
            )

        return distr_at_all_levels

    def _distr_to_samples(
        self,
        distr_at_all_levels: Distribution,
        num_samples: int,
    ):
        if num_samples == 1:
            samples_at_all_levels = distr_at_all_levels.sample(
                num_samples=num_samples, dtype=self.dtype
            )

            # get rid of the redundant axis introduced by `sample`.
            samples_at_all_levels = samples_at_all_levels.squeeze(axis=0)
        else:
            samples_at_all_levels = distr_at_all_levels.sample_rep(
                num_samples=num_samples, dtype=self.dtype
            )

        return samples_at_all_levels


class COPDeepARTrainingNetwork(COPNetwork):
    @validated()
    def __init__(
        self,
        num_batches_per_epoch: int,
        epochs: int,
        warmstart_epoch_frac: float,
        num_samples_for_loss: int = 200,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.epochs = epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.batch_no = 0
        self.num_samples_for_loss = num_samples_for_loss

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Optional[Tensor],
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        agg_features_dict: Dict,
    ) -> Tensor:
        """
        Computes the loss for training COPDeepAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape, seq_len)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, *target_shape)
        future_observed_values : (batch_size, prediction_length, *target_shape)
        agg_features_dict: Dictionary of features for aggregated levels

        Returns loss with shape (batch_size, context + prediction_length, 1)
        -------

        """

        embeddings_at_all_levels_ls = []
        target_at_all_levels_ls = []
        scale_ls = []

        for i, agg_multiple in enumerate(
            self.temporal_hierarchy.agg_multiples
        ):
            if agg_multiple != 1:
                past_time_feat_agg = agg_features_dict[f"level_{i}"][
                    "past_time_feat_agg"
                ]
                future_time_feat_agg = agg_features_dict[f"level_{i}"][
                    "future_time_feat_agg"
                ]
            else:
                past_time_feat_agg = past_time_feat
                future_time_feat_agg = future_time_feat

            target_related_feat_agg = (
                self.get_target_related_feat_at_agg_level(
                    agg_level=i,
                    past_target=past_target,
                    past_is_pad=past_is_pad,
                    past_observed_values=past_observed_values,
                    future_target=future_target,
                    future_observed_values=future_observed_values,
                )
            )

            rnn_outputs, _, scale, _, _ = self.models[i].unroll_encoder(
                F=F,
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_time_feat=past_time_feat_agg,
                future_time_feat=future_time_feat_agg,
                **target_related_feat_agg,
            )
            scale_ls.append(scale.expand_dims(axis=-1))

            # put together target sequence
            # (batch_size, seq_len, *target_shape)
            target = F.concat(
                target_related_feat_agg["past_target"].slice_axis(
                    axis=1,
                    begin=self.models[i].history_length
                    - self.models[i].context_length,
                    end=None,
                ),
                target_related_feat_agg["future_target"],
                dim=1,
            )

            # We reconcile blocks/windows of time steps: e.g., if we have 28 values of daily data, then we
            # reconcile 4 windows where each window has a length of 7 if number of leaves in the hierarchy is 7.
            window_size = self.temporal_hierarchy.num_leaves // agg_multiple
            num_windows = (
                self.models[i].context_length
                + self.models[i].prediction_length
            ) // window_size

            embeddings_at_all_levels_ls.append(
                rnn_outputs.reshape(
                    (
                        rnn_outputs.shape[0],
                        num_windows,
                        -1,
                        rnn_outputs.shape[-1],
                    )
                )
            )

            target_at_all_levels_ls.append(
                target.reshape((target.shape[0], num_windows, -1))
            )

        # Last dimension contains embeddings at all time-levels and message passing/aggregation can be done on it.
        # Shape: (bs, num_windows, total_num_time_steps_of_hierarchy, embedding_dim)
        embeddings_at_all_levels = F.concat(
            *embeddings_at_all_levels_ls, dim=-2
        )

        if self.use_gnn:
            embeddings_at_all_levels = self.gnn(embeddings_at_all_levels)

        distr_at_all_levels = self._embeddings_to_distr(
            F,
            embeddings_at_all_levels,
            scale_ls,
        )

        target_at_all_levels = F.concat(*target_at_all_levels_ls, dim=-1)

        if self.loss_function == "nll":
            loss = distr_at_all_levels.loss(x=target_at_all_levels)

            # Determine which epoch we are currently in.
            self.batch_no += 1
            epoch_no = self.batch_no // self.num_batches_per_epoch + 1
            epoch_frac = epoch_no / self.epochs

            if epoch_frac > self.warmstart_epoch_frac:
                print(
                    f"epoch_frac: {epoch_frac}. Switching the loss function to CRPS"
                )
                self.loss_function = "crps_univariate"

        else:
            samples_at_all_levels = self._distr_to_samples(
                distr_at_all_levels,
                num_samples=self.num_samples_for_loss,
            )

            if self.do_reconciliation:
                reconciled_samples_at_all_levels = reconcile_samples(
                    reconciliation_mat=self.M,
                    samples=samples_at_all_levels,
                    non_negative=self.non_negative,
                )
            else:
                reconciled_samples_at_all_levels = samples_at_all_levels

            loss = (
                EmpiricalDistribution(
                    samples=reconciled_samples_at_all_levels, event_dim=1
                )
                .loss(x=target_at_all_levels)
                .expand_dims(axis=-1)
            )

        return loss


class COPDeepARPredictionNetwork(COPNetwork):
    @validated()
    def __init__(
        self,
        return_forecasts_at_all_levels: bool = False,
        num_parallel_samples: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(prediction=True, **kwargs)
        self.return_forecasts_at_all_levels = return_forecasts_at_all_levels
        self.num_parallel_samples = num_parallel_samples

    def _decode_one_window(
        self,
        F,
        model: DeepARPredictionNetwork,
        window_size: int,
        offset: int,
        static_feat: Tensor,
        past_target: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        begin_states: List,
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes RNN outputs by unrolling the LSTM starting with a initial
        input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length,
            num_time_features).
            Note: They still need to be for all `prediction_length` time steps.
            This function will slice the features it needs.
        scale : Tensor
            tensor containing the scale of each element in the batch.
            Shape: (batch_size, 1, 1).
        begin_states : List
            list of initial states for the LSTM layers. The shape of each
            tensor of the list should be (batch_size, num_cells)
        Returns
        --------
        Tensor
            A tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, window_size).
        """

        rnn_outputs_ls = []

        # for each future time-units we draw new samples for this time-unit and
        # update the state
        for k in range(offset, offset + window_size):
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = model.get_lagged_subsequences(
                F=F,
                sequence=past_target,
                sequence_length=model.history_length + k,
                indices=model.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = F.reshape(
                data=lags_scaled,
                shape=(-1, 1, prod(model.target_shape) * len(model.lags_seq)),
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags +
            # num_time_features + num_static_features)
            decoder_input = F.concat(
                input_lags,
                time_feat.slice_axis(axis=1, begin=k, end=k + 1),
                # observed_values.expand_dims(axis=1),
                static_feat,
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, begin_states = model.rnn.unroll(
                inputs=decoder_input,
                length=1,
                begin_state=begin_states,
                layout="NTC",
                merge_outputs=True,
            )
            rnn_outputs_ls.append(rnn_outputs)

            distr_args = model.proj_distr_args(rnn_outputs)

            # compute likelihood of target given the predicted parameters
            distr = model.distr_output.distribution(distr_args, scale=scale)

            # (batch_size * num_samples, 1, *target_shape)

            new_samples = distr.sample(dtype=self.dtype)

            # (batch_size * num_samples, seq_len, *target_shape)
            past_target = F.concat(past_target, new_samples, dim=1)

        # (batch_size * num_samples, prediction_length, *target_shape)
        rnn_outputs = F.concat(*rnn_outputs_ls, dim=1)

        return rnn_outputs, begin_states

    def sampling_decoder(
        self,
        F,
        state_ls,
        scale_ls,
        static_feat_ls,
        past_target_ls,
        future_time_feat_agg_ls,
    ):
        num_windows = (
            self.prediction_length // self.temporal_hierarchy.num_leaves
        )
        num_nodes_per_level = self.temporal_hierarchy.num_nodes_per_level

        reconciled_samples_at_all_levels_ls = []
        for j in range(num_windows):
            embeddings_at_all_levels_ls = []
            for i, agg_multiple in enumerate(
                self.temporal_hierarchy.agg_multiples
            ):
                rnn_outputs, states = self._decode_one_window(
                    F=F,
                    model=self.models[i],
                    window_size=num_nodes_per_level[i],
                    offset=j * num_nodes_per_level[i],
                    past_target=past_target_ls[i],
                    time_feat=future_time_feat_agg_ls[i],
                    static_feat=static_feat_ls[i],
                    scale=scale_ls[i],
                    begin_states=state_ls[i],
                )

                state_ls[i] = states
                embeddings_at_all_levels_ls.append(
                    rnn_outputs.reshape(
                        (rnn_outputs.shape[0], -1, rnn_outputs.shape[-1])
                    )
                )

            # Last dimension contains embeddings at all time-levels and message passing/aggregation can be done on it.
            # Shape: (bs, total_num_time_steps_of_hierarchy, embedding_dim)
            embeddings_at_all_levels = F.concat(
                *embeddings_at_all_levels_ls, dim=-2
            )

            if self.use_gnn:
                embeddings_at_all_levels = self.gnn(embeddings_at_all_levels)

            distr_at_all_levels = self._embeddings_to_distr(
                F,
                embeddings_at_all_levels,
                scale_ls,
            )

            samples_at_all_levels = self._distr_to_samples(
                distr_at_all_levels,
                num_samples=1,
            )

            if self.do_reconciliation:
                reconciled_samples_at_all_levels = reconcile_samples(
                    reconciliation_mat=self.M,
                    samples=samples_at_all_levels,
                    non_negative=self.non_negative,
                )
            else:
                reconciled_samples_at_all_levels = samples_at_all_levels

            rec_err = coherency_error(
                A=self.A, samples=reconciled_samples_at_all_levels
            )
            print(f"Reconciliation error: {rec_err}")

            cumsum_nodes_per_level = np.cumsum([0] + num_nodes_per_level)
            for i in range(len(self.temporal_hierarchy.agg_multiples)):
                # (batch_size * num_samples, seq_len, *target_shape)
                reconciled_samples = (
                    reconciled_samples_at_all_levels.slice_axis(
                        axis=-1,
                        begin=cumsum_nodes_per_level[i],
                        end=cumsum_nodes_per_level[i + 1],
                    )
                )
                past_target_ls[i] = F.concat(
                    past_target_ls[i], reconciled_samples, dim=1
                )

            reconciled_samples_at_all_levels_ls.append(
                reconciled_samples_at_all_levels.reshape(
                    shape=(
                        -1,
                        self.num_parallel_samples,
                        reconciled_samples_at_all_levels.shape[-1],
                    )
                ).expand_dims(axis=-2)
            )

        reconciled_samples_at_all_levels = F.concat(
            *reconciled_samples_at_all_levels_ls, dim=-2
        )
        print(reconciled_samples_at_all_levels.shape)
        return reconciled_samples_at_all_levels

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        feat_static_real: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Tensor,  # (batch_size, prediction_length, num_features)
        past_is_pad: Tensor,
        agg_features_dict: Dict,
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)
        agg_features_dict: Dictionary of features for aggregated levels

        Returns
        -------
        Tensor
            Predicted samples
        """

        (
            state_ls,
            scale_ls,
            static_feat_ls,
            past_target_ls,
            future_time_feat_agg_ls,
        ) = ([], [], [], [], [])

        for i, agg_multiple in enumerate(
            self.temporal_hierarchy.agg_multiples
        ):
            if agg_multiple != 1:
                past_time_feat_agg = agg_features_dict[f"level_{i}"][
                    "past_time_feat_agg"
                ]
                future_time_feat_agg = agg_features_dict[f"level_{i}"][
                    "future_time_feat_agg"
                ]
            else:
                past_time_feat_agg = past_time_feat
                future_time_feat_agg = future_time_feat

            target_related_feat_agg = (
                self.get_target_related_feat_at_agg_level(
                    agg_level=i,
                    past_target=past_target,
                    past_is_pad=past_is_pad,
                    past_observed_values=past_observed_values,
                )
            )

            # unroll the decoder in "prediction mode", i.e. with past data only
            _, states, scale, static_feat, imputed_sequence = self.models[
                i
            ].unroll_encoder(
                F=F,
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_time_feat=past_time_feat_agg,
                future_observed_values=None,
                future_time_feat=None,
                future_target=None,
                **target_related_feat_agg,
            )

            # blows-up the dimension of each tensor to batch_size *
            # self.num_parallel_samples for increasing parallelism
            repeated_past_target = imputed_sequence.repeat(
                repeats=self.num_parallel_samples, axis=0
            )
            repeated_states = [
                s.repeat(repeats=self.num_parallel_samples, axis=0)
                for s in states
            ]
            repeated_time_feat = future_time_feat_agg.repeat(
                repeats=self.num_parallel_samples, axis=0
            )
            repeated_static_feat = static_feat.repeat(
                repeats=self.num_parallel_samples, axis=0
            ).expand_dims(axis=1)
            repeated_scale = scale.repeat(
                repeats=self.num_parallel_samples, axis=0
            )

            state_ls.append(repeated_states)
            scale_ls.append(repeated_scale)
            static_feat_ls.append(repeated_static_feat)
            past_target_ls.append(repeated_past_target)
            future_time_feat_agg_ls.append(repeated_time_feat)

        reconciled_samples_at_all_levels = self.sampling_decoder(
            F,
            state_ls=state_ls,
            scale_ls=scale_ls,
            static_feat_ls=static_feat_ls,
            past_target_ls=past_target_ls,
            future_time_feat_agg_ls=future_time_feat_agg_ls,
        )

        if self.return_forecasts_at_all_levels:
            return reconciled_samples_at_all_levels
        else:
            reconciled_samples_at_bottom_level = (
                reconciled_samples_at_all_levels.slice_axis(
                    axis=-1,
                    begin=-self.temporal_hierarchy.num_leaves,
                    end=None,
                )
            )

            reconciled_samples_at_bottom_level = (
                reconciled_samples_at_bottom_level.reshape(
                    (
                        reconciled_samples_at_bottom_level.shape[0],
                        reconciled_samples_at_bottom_level.shape[1],
                        -1,
                    )
                )
            )

            return reconciled_samples_at_bottom_level

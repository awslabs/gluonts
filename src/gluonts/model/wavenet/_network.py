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

# Standard library imports
import math
from typing import List, Optional, Tuple

# Third-party imports
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

# First-party imports
from gluonts.block.feature import FeatureEmbedder
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class LookupValues(gluon.HybridBlock):
    def __init__(self, values: mx.nd.NDArray, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.bin_values = self.params.get_constant("bin_values", values)

    def hybrid_forward(self, F, indices, bin_values):
        return F.take(bin_values, indices)


def conv1d(channels, kernel_size, in_channels, use_bias=True, **kwargs):
    """
    Conv1D with better default initialization.
    """
    n = in_channels
    kernel_size = (
        kernel_size if isinstance(kernel_size, list) else [kernel_size]
    )
    for k in kernel_size:
        n *= k
    stdv = 1.0 / math.sqrt(n)
    winit = mx.initializer.Uniform(stdv)
    if use_bias:
        binit = mx.initializer.Uniform(stdv)
    else:
        binit = "zeros"
    return nn.Conv1D(
        channels=channels,
        kernel_size=kernel_size,
        in_channels=in_channels,
        use_bias=use_bias,
        weight_initializer=winit,
        bias_initializer=binit,
        **kwargs,
    )


class CausalDilatedResidue(nn.HybridBlock):
    def __init__(
        self,
        n_residue,
        n_skip,
        dilation,
        return_dense_out,
        kernel_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_residue = n_residue
        self.n_skip = n_skip
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.return_dense_out = return_dense_out
        with self.name_scope():
            self.conv_sigmoid = conv1d(
                in_channels=n_residue,
                channels=n_residue,
                kernel_size=kernel_size,
                dilation=dilation,
                activation="sigmoid",
            )
            self.conv_tanh = conv1d(
                in_channels=n_residue,
                channels=n_residue,
                kernel_size=kernel_size,
                dilation=dilation,
                activation="tanh",
            )
            self.skip = conv1d(
                in_channels=n_residue, channels=n_skip, kernel_size=1
            )
            self.residue = (
                conv1d(
                    in_channels=n_residue, channels=n_residue, kernel_size=1
                )
                if self.return_dense_out
                else None
            )

    def hybrid_forward(self, F, x):
        u = self.conv_sigmoid(x) * self.conv_tanh(x)
        s = self.skip(u)
        if not self.return_dense_out:
            return s, F.zeros(shape=(1,))
        output = self.residue(u)
        output = output + F.slice_axis(
            x, begin=(self.kernel_size - 1) * self.dilation, end=None, axis=-1
        )
        return s, output


class WaveNet(nn.HybridBlock):
    def __init__(
        self,
        bin_values: List[float],
        n_residue: int,
        n_skip: int,
        dilation_depth: int,
        n_stacks: int,
        act_type: str,
        cardinality: List[int],
        embedding_dimension: int,
        pred_length: int,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.dilation_depth = dilation_depth
        self.pred_length = pred_length
        self.bin_values = bin_values
        self.mu = len(bin_values)
        self.dilations = WaveNet._get_dilations(
            dilation_depth=dilation_depth, n_stacks=n_stacks
        )
        self.receptive_field = WaveNet.get_receptive_field(
            dilation_depth=dilation_depth, n_stacks=n_stacks
        )
        self.trim_lengths = [
            sum(self.dilations) - sum(self.dilations[: i + 1])
            for i, _ in enumerate(self.dilations)
        ]

        with self.name_scope():
            self.feature_embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension for _ in cardinality],
            )

            # self.post_transform = LookupValues(mx.nd.array(bin_values))
            self.target_embed = nn.Embedding(
                input_dim=self.mu, output_dim=n_residue
            )
            self.residuals = nn.HybridSequential()
            for i, d in enumerate(self.dilations):
                is_not_last = i + 1 < len(self.dilations)
                self.residuals.add(
                    CausalDilatedResidue(
                        n_residue=n_residue,
                        n_skip=n_skip,
                        dilation=d,
                        return_dense_out=is_not_last,
                        kernel_size=2,
                    )
                )

            std = 1.0 / math.sqrt(n_residue)
            self.conv_project = nn.Conv1D(
                channels=n_residue,
                kernel_size=1,
                use_bias=True,
                weight_initializer=mx.init.Uniform(std),
                bias_initializer="zero",
            )

            self.conv1 = conv1d(
                in_channels=n_skip, channels=n_skip, kernel_size=1
            )

            self.conv2 = conv1d(
                in_channels=n_skip, channels=self.mu, kernel_size=1
            )
            self.output_act = (
                nn.ELU()
                if act_type == "elu"
                else nn.Activation(activation=act_type)
            )
            self.cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    @staticmethod
    def _get_dilations(dilation_depth, n_stacks):
        return [2 ** i for i in range(dilation_depth)] * n_stacks

    @staticmethod
    def get_receptive_field(dilation_depth, n_stacks):
        """
        Return the length of the receptive field
        """
        dilations = WaveNet._get_dilations(
            dilation_depth=dilation_depth, n_stacks=n_stacks
        )
        return sum(dilations) + 1

    def is_last_layer(self, i):
        return i + 1 == len(self.dilations)

    def get_full_features(
        self,
        F,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        future_observed_values: Optional[Tensor],
        scale: Tensor,
    ):
        """
        Prepares the inputs for the network by repeating static feature and concatenating it with time features and
        observed value indicator.

        Parameters
        ----------
        F
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        past_observed_values
            Observed value indicator for the past target: (batch_size, receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features, receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        future_observed_values
            Observed value indicator for the future target: (batch_size, pred_length).
            This will be set to all ones, if not provided (e.g., during prediction).
        scale
            scale of the time series: (batch_size, 1)
        Returns
        -------
        Tensor
            A tensor containing all the features ready to be passed through the network.
            Shape: (batch_size, num_features, receptive_field + pred_length)
        """
        embedded_cat = self.feature_embedder(feat_static_cat)
        static_feat = F.concat(embedded_cat, F.log(scale + 1.0), dim=1)
        repeated_static_feat = F.repeat(
            F.expand_dims(static_feat, axis=-1),
            repeats=self.pred_length + self.receptive_field,
            axis=-1,
        )

        if future_observed_values is None:
            future_observed_values = (
                F.slice_axis(future_time_feat, begin=0, end=1, axis=1)
                .squeeze(axis=1)
                .ones_like()
            )
        full_observed = F.expand_dims(
            F.concat(past_observed_values, future_observed_values, dim=-1),
            axis=1,
        )
        full_time_features = F.concat(past_time_feat, future_time_feat, dim=-1)
        full_features = F.concat(
            full_time_features, full_observed, repeated_static_feat, dim=1
        )
        return full_features

    def target_feature_embedding(
        self, F, target, features,
    ):
        """
        Provides a joint embedding for the target and features.

        Parameters
        ----------
        F
        target: (batch_size, sequence_length)
        features: (batch_size, num_features, sequence_length)

        Returns
        -------
        Tensor
            A tensor containing a joint embedding of target and features.
            Shape: (batch_size, n_residue, sequence_length)

        """
        # (batch_size, embed_dim, sequence_length)
        o = self.target_embed(target).swapaxes(1, 2)
        o = F.concat(o, features, dim=1)
        o = self.conv_project(o)
        return o

    def base_net(
        self, F, inputs, one_step_prediction=False, queues=None
    ) -> Tuple[Tensor, List]:
        """
        Forward pass through the network.

        Parameters
        ----------
        F
        inputs
            Inputs to the network: (batch_size, n_residue, sequence_length)
        one_step_prediction
            Flag indicating whether the network is "unrolled/propagated" one step at a time (prediction phase).
        queues
            Convolutional queues containing past computations. Should be provided if `one_step_prediction` is True.

        Returns
        -------
        Tuple: (Tensor, List)
            A tensor containing the unnormalized outputs of the network. Shape: (batch_size, pred_length, num_bins).
            A list containing the convolutional queues for each layer. The queue corresponding to layer `l` has
            shape: (batch_size, n_residue, 2^l).
        """
        if one_step_prediction:
            assert (
                queues is not None
            ), "Queues must not be empty during prediction phase!"
        skip_outs = []
        queues_next = []
        o = inputs
        for i, d in enumerate(self.dilations):
            skip, o = self.residuals[i](o)
            if one_step_prediction:
                skip_trim = skip
                if not self.is_last_layer(i):
                    q = queues[i]
                    o = F.concat(q, o, num_args=2, dim=-1)
                    queues_next.append(
                        F.slice_axis(o, begin=1, end=None, axis=-1)
                    )
            else:
                skip_trim = F.slice_axis(
                    skip, begin=self.trim_lengths[i], end=None, axis=-1
                )
            skip_outs.append(skip_trim)
        y = sum(skip_outs)
        y = self.output_act(y)
        y = self.conv1(y)
        y = self.output_act(y)
        y = self.conv2(y)
        unnormalized_output = y.swapaxes(1, 2)
        return unnormalized_output, queues_next

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        scale: Tensor,
    ) -> Tensor:
        """
        Computes the training loss for the wavenet model.

        Parameters
        ----------
        F
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        past_target
            Past target: (batch_size, receptive_field)
        past_observed_values
            Observed value indicator for the past target: (batch_size, receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features, receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        future_target
            Target on which the loss is computed: (batch_size, pred_length)
        future_observed_values
            Observed value indicator for the future target: (batch_size, pred_length).
        scale
            scale of the time series: (batch_size, 1)

        Returns
        -------
        Tensor
            Returns loss with shape (batch_size,)
        """
        full_target = F.concat(past_target, future_target, dim=-1).astype(
            "int32"
        )
        full_features = self.get_full_features(
            F,
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_observed_values=future_observed_values,
            scale=scale,
        )
        embedding = self.target_feature_embedding(
            F,
            F.slice_axis(full_target, begin=0, end=-1, axis=-1),
            F.slice_axis(full_features, begin=1, end=None, axis=-1),
        )
        unnormalized_output, _ = self.base_net(F, embedding)

        label = F.slice_axis(
            full_target, begin=self.receptive_field, end=None, axis=-1
        )

        full_observed = F.expand_dims(
            F.concat(past_observed_values, future_observed_values, dim=-1),
            axis=1,
        )
        loss_weight = F.slice_axis(
            full_observed, begin=self.receptive_field, end=None, axis=-1
        )
        loss_weight = F.expand_dims(loss_weight, axis=2)
        loss = self.cross_entropy_loss(unnormalized_output, label, loss_weight)
        return loss


class WaveNetPredictor(WaveNet):
    """
    Base class for wavenet prediction. Runs Wavenet generation in an auto-regressive manner using caching for
    speedup [PKC+16]_.

    Same arguments as WaveNet. In addition

    Parameters
    ----------
    temperature
        If set to 1.0 (default), sample according to estimated probabilities, if set to 0.0
        most likely sample at each step is chosen.
    """

    @validated()
    def __init__(
        self, temperature: float = 1.0, **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature

    def get_initial_conv_queues(self, F, past_target, features):
        """
        Build convolutional queues saving intermediate computations.

        Parameters
        ----------
        F
        past_target: (batch_size, receptive_field)
        features: (batch_size, num_features, receptive_field)

        Returns
        -------
        List
            A list containing the convolutional queues for each layer. The queue corresponding to layer `l` has
            shape: (batch_size, n_residue, 2^l).
        """
        o = self.target_feature_embedding(F, past_target, features)

        queues = []
        for i, d in enumerate(self.dilations):
            sz = 1 if d == 2 ** (self.dilation_depth - 1) else d * 2
            _, o = self.residuals[i](o)
            if not self.is_last_layer(i):
                o_chunk = F.slice_axis(o, begin=-sz - 1, end=-1, axis=-1)
            else:
                o_chunk = o
            queues.append(o_chunk)
        return queues

    def _predict_one_step(self, F, past_target, features, queues):
        """
        Computes prediction for one step.

        Parameters
        ----------
        F
        past_target
            Past target with shape (batch_size, 2)
        features
            Features with shape (batch_size, num_features, 2)
        queues
            List of convolutional queues for each layer. The queue corresponding to layer `l` has
            shape: (batch_size, n_residue, 2^l).

        Returns
        -------
        Tuple containing
            a prediction sample for the next time step with shape (batch_size, 1)
            bin probabilities of the forecast distribution for the next time step with shape (batch_size, 1, num_bins)
            queues updated after propagating the network for one time step.

        Note: `temperature` only affects the way samples are drawn at each time step (as input to the next time step).
        The bin probabilities returned are not affected by `temperature`.

        """
        embedding = self.target_feature_embedding(
            F, target=past_target, features=features,
        )

        # (batch_size, 1, num_bins) where 1 corresponds to the time axis.
        unnormalized_outputs, queues = self.base_net(
            F, embedding, one_step_prediction=True, queues=queues
        )
        # (batch_size, 1, num_bins) where 1 corresponds to the time axis.
        bin_probs = F.softmax(unnormalized_outputs, axis=-1)
        if self.temperature > 0:
            # (batch_size, 1, num_bins) where 1 corresponds to the time axis.
            probs = F.softmax(unnormalized_outputs / self.temperature, axis=-1)
            # (batch_size, 1)
            sample = F.sample_multinomial(probs)
        else:
            # (batch_size, 1)
            sample = F.argmax(unnormalized_outputs, axis=-1)

        return sample, bin_probs, queues

    def predict(
        self,
        F,
        feat_static_cat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        scale: Tensor,
        num_samples: int,
        sample_based_forecast: bool,
    ) -> Tensor:
        """
        Computes predictions from the trained wavenet model.

        Parameters
        ----------
        F
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        past_target
            Past target: (batch_size, receptive_field)
        past_observed_values
            Observed value indicator for the past target: (batch_size, receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features, receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        scale
            scale of the time series: (batch_size, 1)
        num_samples
            Number of samples to draw at each time step in the prediction range.
        sample_based_forecast
            Flag to indicate whether to output sample forecasts or bin probabilities of the forecast distribution.

        Returns
        -------
        Tensor
            Prediction samples with shape (batch_size, num_samples, pred_length), if `sample_based_forecast` = True,
            Bin probabilities for the prediction range with shape (batch_size, pred_length, num_bins), otherwise.
        """
        if not sample_based_forecast:
            assert (
                num_samples == 1
            ), "Number of samples must be 1 since the forecast is not sample based."

        def blow_up(u):
            """
            Expand to (batch_size x num_samples)
            """
            return F.repeat(u, repeats=num_samples, axis=0)

        past_target = past_target.astype("int32")
        full_features = self.get_full_features(
            F,
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_observed_values=None,
            scale=scale,
        )

        # To compute queues for the first step, we need features from
        # -self.pred_length - self.receptive_field + 1 to -self.pred_length + 1
        features_end_ix = (
            -self.pred_length + 1 if self.pred_length > 1 else None
        )
        queues = self.get_initial_conv_queues(
            F,
            past_target=F.slice_axis(
                past_target, begin=-self.receptive_field, end=None, axis=-1
            ),
            features=F.slice_axis(
                full_features,
                begin=-self.pred_length - self.receptive_field + 1,
                end=features_end_ix,
                axis=-1,
            ),
        )
        queues = [blow_up(queue) for queue in queues]

        res = F.slice_axis(past_target, begin=-2, end=None, axis=-1)
        res = blow_up(res)

        future_bin_probs = []
        for n in range(self.pred_length):
            # Generate one-step ahead predictions. The input consists of target and features
            # corresponding to the last two time steps.
            past_target = F.slice_axis(res, begin=-2, end=None, axis=-1)
            current_features = F.slice_axis(
                full_features,
                begin=self.receptive_field + n - 1,
                end=self.receptive_field + n + 1,
                axis=-1,
            )
            sample, bin_probs, queues = self._predict_one_step(
                F,
                past_target=past_target,
                features=blow_up(current_features),
                queues=queues,
            )
            sample = sample.astype("int32")
            res = F.concat(res, sample, num_args=2, dim=-1)
            if not sample_based_forecast:
                future_bin_probs.append(bin_probs)
        if sample_based_forecast:
            samples = F.slice_axis(
                res, begin=-self.pred_length, end=None, axis=-1
            )
            samples = samples.reshape(
                shape=(-1, num_samples, self.pred_length)
            )
            return samples
        else:
            return F.concat(*future_bin_probs, dim=1)


class WaveNetSampler(WaveNetPredictor):
    """
    Runs Wavenet generation in an auto-regressive manner using caching for
    speedup [PKC+16]_ to produce sample forecast.

    Same arguments as WaveNet. In addition

    Parameters
    ----------
    num_samples
        Number of sample paths to generate in parallel in the graph
    temperature
        If set to 1.0 (default), sample according to estimated probabilities, if set to 0.0
        most likely sample at each step is chosen.
    """

    @validated()
    def __init__(
        self, num_samples: int, temperature: float = 1.0, **kwargs,
    ):
        super().__init__(temperature=temperature, **kwargs)
        self.num_samples = num_samples

        with self.name_scope():
            self.post_transform = LookupValues(mx.nd.array(self.bin_values))

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        scale: Tensor,
    ) -> Tensor:
        """
        Generates prediction samples from the trained wavenet model.

        Parameters
        ----------
        F
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        past_target
            Past target: (batch_size, receptive_field)
        past_observed_values
            Observed value indicator for the past target: (batch_size, receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features, receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        scale
            scale of the time series: (batch_size, 1)

        Returns
        -------
        Tensor
            Prediction samples with shape (batch_size, num_samples, pred_length)
        """

        samples_bin_ix = self.predict(
            F,
            feat_static_cat=feat_static_cat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            scale=scale,
            num_samples=self.num_samples,
            sample_based_forecast=True,
        )
        samples = self.post_transform(samples_bin_ix)
        samples = F.broadcast_mul(scale.expand_dims(axis=1), samples)
        return samples


class WaveNetDistributionPredictor(WaveNetPredictor):
    """
    Runs Wavenet generation in an auto-regressive manner using caching for
    speedup [PKC+16]_ to produce (parametric) distribution forecast.

    Same arguments as `WaveNetPredictor`.
    """

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        scale: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor]:
        """
        Computes the forecast distribution from the trained wavenet model.

        Parameters
        ----------
        F
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        past_target
            Past target: (batch_size, receptive_field)
        past_observed_values
            Observed value indicator for the past target: (batch_size, receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features, receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        scale
            scale of the time series: (batch_size, 1)

        Returns
        -------
        Tuple containing `dist_args`, `loc` and `scale`.

            `dist_args` is a pair of tensors containing logarithm of bin probabilities with shape
            (batch_size, pred_length, num_bins) and bin centers with shape (batch_size, num_bins).

            `loc` is a Tensor with shape (batch_size, 1).

            `scale` is a Tensor with shape (batch_size, 1).
        """

        bin_probs = self.predict(
            F,
            feat_static_cat=feat_static_cat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            scale=scale,
            num_samples=1,
            sample_based_forecast=False,
        )

        # get the batch shape right
        scale = F.broadcast_add(scale, F.zeros_like(feat_static_cat))
        loc = F.zeros_like(scale)
        bin_centers = F.broadcast_add(
            F.array(self.bin_values), F.zeros_like(feat_static_cat)
        )

        dist_args = F.log(bin_probs), bin_centers
        return dist_args, loc, scale

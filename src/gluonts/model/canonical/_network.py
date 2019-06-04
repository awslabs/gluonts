# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.block.feature import FeatureEmbedder
from gluonts.block.scaler import MeanScaler
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput
from gluonts.model.common import Tensor


class CanonicalNetworkBase(HybridBlock):
    @validated()
    def __init__(
        self,
        model: HybridBlock,
        embedder: FeatureEmbedder,
        distr_output: DistributionOutput,
        is_sequential: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.distr_output = distr_output
        self.is_sequential = is_sequential
        self.model = model
        self.embedder = embedder

        with self.name_scope():
            self.proj_distr_args = self.distr_output.get_args_proj()
            self.scaler = MeanScaler()

    def assemble_features(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        time_feat: Tensor,  # (batch_size, history_length, num_features)
    ) -> Tensor:
        embedded_cat = self.embedder(
            feat_static_cat
        )  # (batch_size, num_features * embedding_size)

        # a workaround when you wish to repeat without knowing the number
        # of repeats
        helper_ones = F.ones_like(
            F.slice_axis(time_feat, axis=2, begin=-1, end=None)
        )
        # (batch_size, history_length, num_features * embedding_size)
        repeated_cat = F.batch_dot(
            helper_ones, F.expand_dims(embedded_cat, axis=1)
        )

        # putting together all the features
        input_feat = F.concat(repeated_cat, time_feat, dim=2)
        return input_feat

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError


class CanonicalTrainingNetwork(CanonicalNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,
        # (batch_size, num_features, history_length)
        past_target: Tensor,  # (batch_size, history_length)
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space
        feat_static_cat
            Shape: (batch_size, num_features)
        past_time_feat
            Shape: (batch_size, history_length, num_features)
        past_target
            Shape: (batch_size, history_length)

        Returns
        -------
        Tensor
            A batch of negative log likelihoods.
        """
        _, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),  # TODO: pass the actual observed here
        )

        input_feat = self.assemble_features(F, feat_static_cat, past_time_feat)
        outputs = self.model(input_feat)

        distr = self.distr_output.distribution(
            self.proj_distr_args(outputs),
            scale=target_scale.expand_dims(axis=1).expand_dims(axis=2),
        )

        loss = distr.loss(past_target.expand_dims(axis=-1))

        return loss


class CanonicalPredictionNetwork(CanonicalNetworkBase):
    @validated()
    def __init__(
        self, prediction_len: int, num_sample_paths: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_len = prediction_len
        self.num_sample_paths = num_sample_paths

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        past_target: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space module.
        feat_static_cat
            Shape: (batch_size, num_features).
        past_time_feat
            Shape: (batch_size, history_length, num_features).
        future_time_feat
            Shape: (batch_size, history_length, num_features).
        past_target
            Shape: (batch_size, history_length).

        Returns
        -------
        Tensor
            a batch of prediction samples
            Shape: (batch_size, prediction_length, num_sample_paths)
        """

        _, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),  # TODO: pass the actual observed here
        )

        time_feat = (
            F.concat(past_time_feat, future_time_feat, dim=1)
            if self.is_sequential
            else future_time_feat
        )

        input_feat = self.assemble_features(F, feat_static_cat, time_feat)
        outputs = self.model(input_feat)

        if self.is_sequential:
            outputs = F.slice_axis(
                outputs, axis=1, begin=-self.prediction_len, end=None
            )

        distr = self.distr_output.distribution(
            self.proj_distr_args(outputs),
            target_scale.expand_dims(axis=1).expand_dims(axis=2),
        )
        samples = distr.sample(
            self.num_sample_paths
        )  # (num_samples, batch_size, prediction_length, 1)
        return samples.swapaxes(0, 1).squeeze(axis=-1)

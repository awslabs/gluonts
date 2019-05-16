# Standard library imports
from typing import List

# Third-party imports
from mxnet import gluon
from mxnet.gluon import nn
from pydantic import conint, constr

# First-party imports
from gluonts.block.feature import FeatureAssembler
from gluonts.core.component import ConfigBase, validated
from gluonts.distribution import DistributionOutput
from gluonts.model.common import Tensor


class MLPLayerConfig(ConfigBase):
    units: conint(gt=0, le=10000)
    activation: constr(
        regex='linear|relu|sigmoid|tanh|softrelu|softsign'
    ) = 'relu'


class MLPNetworkBase(gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        layer_configs: List[MLPLayerConfig],
        prediction_length: int,
        assemble_feature: FeatureAssembler,
        distr_output: DistributionOutput,
        num_samples: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.prediction_length = prediction_length
        self.assemble_feature = assemble_feature

        self.distr_output = distr_output
        self.num_samples = num_samples

        with self.name_scope():
            self.distr_args_proj = distr_output.get_args_proj()
            self.hybrid_sequential = nn.HybridSequential()
            for layer_no, config in enumerate(layer_configs):
                layer = gluon.nn.Dense(
                    units=config.units,
                    activation=None
                    if config.activation == 'linear'
                    else config.activation,
                    prefix=f'fc_{layer_no:02d}_',
                    flatten=False,
                )
                self.hybrid_sequential.add(layer)

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError


class MLPTrainingNetwork(MLPNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        past_feat_dynamic_real: Tensor,
    ) -> Tensor:
        # 1) assemble features into one big tensor
        features = self.assemble_feature(
            feat_static_cat,
            feat_static_real,
            past_feat_dynamic_cat,
            past_feat_dynamic_real,
        )
        # 2) pass through your everyday MLP
        mlp_outputs = self.hybrid_sequential(features)
        # 3) project latent distribution parameters
        distr_args = self.distr_args_proj(mlp_outputs)
        # 4) calculate loss
        distr = self.distr_output.distribution(distr_args)

        return distr.loss(past_target)


class MLPPredictionNetwork(MLPNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        past_feat_dynamic_real: Tensor,
    ) -> List[Tensor]:
        # 1) assemble features into one big tensor
        features = self.assemble_feature(
            feat_static_cat,
            feat_static_real,
            past_feat_dynamic_cat,
            past_feat_dynamic_real,
        )
        # 2) pass through your everyday MLP
        mlp_outputs = self.hybrid_sequential(
            features
        )  # num_ts x prediction_length x num_feat
        # 3) project latent distribution parameters
        distr_args = self.distr_args_proj(mlp_outputs)
        # 4) generate samples
        distr = self.distr_output.distribution(distr_args)
        return distr.sample(num_samples=self.num_samples).swapaxes(0, 1)

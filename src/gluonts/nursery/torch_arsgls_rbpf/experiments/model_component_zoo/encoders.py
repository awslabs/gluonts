from box import Box
import torch
from torch import nn
from torch.distributions import OneHotCategorical, MultivariateNormal

from torch_extensions.distributions.conditional_parametrised_distribution import \
    ParametrisedConditionalDistribution, \
    LadderParametrisedConditionalDistribution
from torch_extensions.layers_with_init import Linear, Conv2d
from torch_extensions.mlp import MLP
from torch_extensions.ops import batch_diag_matrix
from utils.utils import SigmoidLimiter, Lambda, \
    compute_cnn_output_filters_and_dims, Reshape, \
    IndependentNormal


def _extract_dims_from_cfg_obs(config):
    if config.dims.ctrl_switch is not None:
        dim_in = config.dims.obs + config.dims.ctrl_switch
    else:
        dim_in = config.dims.obs
    dim_out = config.dims.switch
    dims_stem = config.obs_to_switch_encoder_dims
    activations_stem = config.obs_to_switch_encoder_activations
    dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in
    return dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params


def _extract_dims_from_cfg_state(config):
    dim_sufficient_stats_state = int(
        config.dims.state
        + config.dims.state
        + (config.dims.state ** 2 - config.dims.state) / 2
    )
    if config.dims.ctrl_switch is not None:
        dim_in = dim_sufficient_stats_state + config.dims.ctrl_switch
    else:
        dim_in = dim_sufficient_stats_state
    dim_out = config.dims.switch
    dims_stem = config.state_to_switch_encoder_dims
    activations_stem = config.state_to_switch_encoder_activations
    dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in
    return dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params


class ObsToSwitchEncoderCategoricalMLP(ParametrisedConditionalDistribution):
    def __init__(self, config):
        dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params = \
            _extract_dims_from_cfg_obs(config=config)
        super().__init__(
            stem=MLP(dim_in=dim_in,
                     dims_hidden=dims_stem,
                     activations=activations_stem),
            dist_params=nn.ModuleDict({
                "logits": nn.Sequential(
                    Linear(in_features=dim_in_dist_params,
                           out_features=dim_out),
                    SigmoidLimiter(limits=[-5, 5]),
                )
            }),
            dist_cls=OneHotCategorical,
        )


class ObsToSwitchEncoderGaussianMLP(ParametrisedConditionalDistribution):
    def __init__(self, config):
        dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params = \
            _extract_dims_from_cfg_obs(config=config)
        super().__init__(
            stem=MLP(dim_in=dim_in,
                     dims_hidden=dims_stem,
                     activations=activations_stem),
            dist_params=nn.ModuleDict({
                "loc": nn.Sequential(
                    Linear(in_features=dim_in_dist_params,
                           out_features=dim_out),
                ),
                "scale_tril": nn.Sequential(
                    Linear(dim_in_dist_params, dim_out),
                    Lambda(fn=lambda x: x - 2),
                    nn.Softplus(),
                    Lambda(fn=lambda x: x + 1e-6),
                    Lambda(fn=batch_diag_matrix),
                ),
            }),
            dist_cls=MultivariateNormal,
        )


class StateToSwitchEncoderCategoricalMLP(ParametrisedConditionalDistribution):
    def __init__(self, config):
        dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params = \
            _extract_dims_from_cfg_state(config=config)
        super().__init__(
            stem=MLP(dim_in=dim_in,
                     dims_hidden=dims_stem,
                     activations=activations_stem),
            dist_params=nn.ModuleDict({
                "logits": nn.Sequential(
                    Linear(in_features=dim_in_dist_params,
                           out_features=dim_out),
                    SigmoidLimiter(limits=[-5, 5]),
                )
            }),
            dist_cls=OneHotCategorical,
        )


class StateToSwitchEncoderGaussianMLP(ParametrisedConditionalDistribution):
    def __init__(self, config):
        dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params = \
            _extract_dims_from_cfg_state(config=config)
        super().__init__(
            stem=MLP(dim_in=dim_in,
                     dims_hidden=dims_stem,
                     activations=activations_stem),
            dist_params=nn.ModuleDict({
                "loc": nn.Sequential(
                    Linear(in_features=dim_in_dist_params,
                           out_features=dim_out),
                    SigmoidLimiter(limits=[-5, 5]),
                ),
                "scale_tril": nn.Sequential(
                    Linear(dim_in_dist_params, dim_out),
                    nn.Softplus(),
                    Lambda(fn=lambda x: x + 1e-6),
                    Lambda(fn=batch_diag_matrix),
                ),
            }),
            dist_cls=MultivariateNormal,
        )


class ObsToSwitchEncoderConvCategorical(ParametrisedConditionalDistribution):
    """ Identical stem architecture as KVAE observation encoder - only distribution is different """

    def __init__(self, config):
        shp_enc_out, dim_out_flat_conv = compute_cnn_output_filters_and_dims(
            dims_img=config.dims_img,
            dims_filter=config.dims_conv,
            kernel_sizes=config.kernel_sizes_conv,
            strides=config.kernel_sizes_conv,
            paddings=config.paddings_conv,
        )
        super().__init__(
            stem=nn.Sequential(
                Reshape(config.dims_img),  # TxPxB will be flattened before.
                Conv2d(in_channels=config.dims_img[0],
                       out_channels=config.dims_conv[0],
                       kernel_size=config.kernel_sizes_conv[0],
                       stride=config.strides_conv[0],
                       padding=config.paddings_conv[0]),
                nn.ReLU(),
                Conv2d(in_channels=config.dims_conv[0],
                       out_channels=config.dims_conv[1],
                       kernel_size=config.kernel_sizes_conv[1],
                       stride=config.strides_conv[1],
                       padding=config.paddings_conv[1]),
                nn.ReLU(),
                Conv2d(in_channels=config.dims_conv[1],
                       out_channels=config.dims_conv[2],
                       kernel_size=config.kernel_sizes_conv[2],
                       stride=config.strides_conv[2],
                       padding=config.paddings_conv[2]),
                nn.ReLU(),
                Reshape((dim_out_flat_conv,)),  # Flatten image dims
            ),
            dist_params=nn.ModuleDict({
                "logits": Linear(in_features=dim_out_flat_conv,
                                 out_features=config.dims.switch),
            }),
            dist_cls=OneHotCategorical,
        )


class ObsToAuxiliaryEncoderConvGaussian(ParametrisedConditionalDistribution):
    def __init__(self, config):
        shp_enc_out, dim_out_flat_conv = compute_cnn_output_filters_and_dims(
            dims_img=config.dims_img,
            dims_filter=config.dims_filter,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            paddings=config.paddings,
        )
        super().__init__(
            stem=nn.Sequential(
                Reshape(config.dims_img),  # TxPxB will be flattened before.
                nn.ZeroPad2d(padding=[0, 1, 0, 1]),
                Conv2d(in_channels=config.dims_img[0],
                       out_channels=config.dims_filter[0],
                       kernel_size=config.kernel_sizes[0],
                       stride=config.strides[0],
                       padding=0),
                nn.ReLU(),
                nn.ZeroPad2d(padding=[0, 1, 0, 1]),
                Conv2d(in_channels=config.dims_filter[0],
                       out_channels=config.dims_filter[1],
                       kernel_size=config.kernel_sizes[1],
                       stride=config.strides[1],
                       padding=0),
                nn.ReLU(),
                nn.ZeroPad2d(padding=[0, 1, 0, 1]),
                Conv2d(in_channels=config.dims_filter[1],
                       out_channels=config.dims_filter[2],
                       kernel_size=config.kernel_sizes[2],
                       stride=config.strides[2],
                       padding=0),
                nn.ReLU(),
                Reshape((dim_out_flat_conv,)),  # Flatten image dims
            ),
            dist_params=nn.ModuleDict({
                "loc": Linear(in_features=dim_out_flat_conv,
                              out_features=config.dims.auxiliary),
                "scale": nn.Sequential(
                    Linear(in_features=dim_out_flat_conv,
                           out_features=config.dims.auxiliary),
                    Lambda(fn=lambda x: torch.sqrt(
                        (config.init_scale_Q_diag ** 2) * nn.functional.sigmoid(
                            x))),
                ),
            }),
            dist_cls=IndependentNormal,
        )


class ObsToAuxiliaryEncoderMlpGaussian(ParametrisedConditionalDistribution):
    def __init__(self, config):
        dim_in = config.dims.obs
        dim_out = config.dims.auxiliary
        dims_stem = config.dims_encoder
        activations_stem = config.activations_decoder
        dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in

        super().__init__(
            stem=MLP(dim_in=dim_in,
                     dims_hidden=dims_stem,
                     activations=activations_stem),
            dist_params=nn.ModuleDict({
                "loc": nn.Sequential(
                    Linear(in_features=dim_in_dist_params,
                           out_features=dim_out),
                ),
                "scale_tril": nn.Sequential(
                    Linear(dim_in_dist_params, dim_out),
                    Lambda(fn=lambda x: x - 2),
                    # start with smaller scale to reduce noise early.
                    nn.Softplus(),
                    Lambda(fn=lambda x: x + 1e-6),
                    Lambda(fn=batch_diag_matrix),
                ),
            }),
            dist_cls=MultivariateNormal,
        )


class ObsToAuxiliaryLadderEncoderMlpGaussian(
    LadderParametrisedConditionalDistribution):
    def __init__(self, config):
        self.num_hierarchies = 2
        # TODO: which control to take?
        assert config.dims.ctrl_switch == config.dims.ctrl_obs == config.dims.ctrl_state

        dim_in = config.dims.obs + config.dims.ctrl_switch
        dim_out_1 = config.dims.auxiliary
        dim_out_2 = config.dims.switch

        dims_stems = config.dims_encoder
        activations_stems = config.activations_encoders
        dim_in_stem_2 = dims_stems[0][-1] if len(dims_stems[0]) > 0 else dim_in
        dim_in_dist_params_1 = dims_stems[0][-1] if len(
            dims_stems[0]) > 0 else dim_in
        dim_in_dist_params_2 = dims_stems[1][-1] if len(
            dims_stems[1]) > 0 else dim_in_stem_2

        super().__init__(
            stems=nn.ModuleList([
                MLP(dim_in=dim_in, dims_hidden=dims_stems[0],
                    activations=activations_stems[0]),
                MLP(dim_in=dim_in_stem_2, dims_hidden=dims_stems[1],
                    activations=activations_stems[1]),
            ]),
            dist_params=nn.ModuleList([
                nn.ModuleDict({
                    "loc": nn.Sequential(
                        Linear(in_features=dim_in_dist_params_1,
                               out_features=dim_out_1),
                    ),
                    "scale_tril": nn.Sequential(
                        Linear(dim_in_dist_params_1, dim_out_1),
                        Lambda(fn=lambda x: x - 2),
                        nn.Softplus(),
                        Lambda(fn=lambda x: x + 1e-6),
                        Lambda(fn=batch_diag_matrix),
                    ),
                }),
                nn.ModuleDict({
                    "loc": nn.Sequential(
                        Linear(in_features=dim_in_dist_params_2,
                               out_features=dim_out_2),
                    ),
                    "scale_tril": nn.Sequential(
                        Linear(dim_in_dist_params_2, dim_out_2),
                        Lambda(fn=lambda x: x - 2),
                        nn.Softplus(),
                        Lambda(fn=lambda x: x + 1e-6),
                        Lambda(fn=batch_diag_matrix),
                    ),
                }),
            ]),
            dist_cls=[MultivariateNormal, MultivariateNormal],
        )

    def forward(self, *args, **kwargs):
        dists = super().forward(*args, **kwargs)
        assert len(dists) == self.num_hierarchies and \
               all(isinstance(dist, torch.distributions.Distribution) for dist
                   in dists)
        return Box(auxiliary=dists[0], switch=dists[1])


class ObsToAuxiliaryLadderEncoderConvMlpGaussian(
    LadderParametrisedConditionalDistribution):
    def __init__(self, config):
        self.num_hierarchies = 2
        assert config.dims.ctrl_switch \
               == config.dims.ctrl_obs \
               == config.dims.ctrl_state \
               in [None, 0], \
            "no controls. would require different architecture or mixing with images."
        shp_enc_out, dim_out_flat_conv = compute_cnn_output_filters_and_dims(
            dims_img=config.dims_img,
            dims_filter=config.dims_filter,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            paddings=config.paddings,
        )

        assert config.dims_encoder[0] is None, "first stem is a conv net. " \
                                               "config is given differently..."
        dims_stem_2 = (32, 32,)
        activations_stem_2 = nn.ReLU()
        dim_out_1 = config.dims.auxiliary
        dim_out_2 = config.dims.switch
        dim_in_dist_params_1 = dim_out_flat_conv
        dim_in_dist_params_2 = dims_stem_2[-1] if len(
            dims_stem_2) > 0 else dim_out_flat_conv

        super().__init__(
            stems=nn.ModuleList([
                nn.Sequential(
                    Reshape(config.dims_img),  # TxPxB will be flattened before.
                    Conv2d(in_channels=config.dims_img[0],
                           out_channels=config.dims_filter[0],
                           kernel_size=config.kernel_sizes[0],
                           stride=config.strides[0],
                           padding=config.paddings[0]),
                    nn.ReLU(),
                    Conv2d(in_channels=config.dims_filter[0],
                           out_channels=config.dims_filter[1],
                           kernel_size=config.kernel_sizes[1],
                           stride=config.strides[1],
                           padding=config.paddings[1]),
                    nn.ReLU(),
                    Conv2d(in_channels=config.dims_filter[1],
                           out_channels=config.dims_filter[2],
                           kernel_size=config.kernel_sizes[2],
                           stride=config.strides[2],
                           padding=config.paddings[2]),
                    nn.ReLU(),
                    Reshape((dim_out_flat_conv,)),  # Flatten image dims
                ),
                MLP(
                    dim_in=dim_out_flat_conv,
                    dims_hidden=dims_stem_2,
                    activations=activations_stem_2
                ),
            ]),
            dist_params=nn.ModuleList([
                nn.ModuleDict({
                    "loc": nn.Sequential(
                        Linear(in_features=dim_in_dist_params_1,
                               out_features=dim_out_1),
                    ),
                    "scale_tril": nn.Sequential(
                        Linear(dim_in_dist_params_1, dim_out_1),
                        Lambda(fn=lambda x: x - 2),
                        nn.Softplus(),
                        Lambda(fn=lambda x: x + 1e-6),
                        Lambda(fn=batch_diag_matrix),
                    ),
                }),
                nn.ModuleDict({
                    "loc": nn.Sequential(
                        Linear(in_features=dim_in_dist_params_2,
                               out_features=dim_out_2),
                    ),
                    "scale_tril": nn.Sequential(
                        Linear(dim_in_dist_params_2, dim_out_2),
                        # Lambda(fn=lambda x: torch.sqrt(  # as in KVAE
                        #     (config.init_scale_Q_diag ** 2) * nn.functional.sigmoid(x))),
                        Lambda(fn=lambda x: x - 2),
                        nn.Softplus(),
                        Lambda(fn=lambda x: x + 1e-6),
                        Lambda(fn=batch_diag_matrix),
                    ),
                }),
            ]),
            dist_cls=[MultivariateNormal, MultivariateNormal],
        )

    def forward(self, *args, **kwargs):
        dists = super().forward(*args, **kwargs)
        assert len(dists) == self.num_hierarchies and \
               all(isinstance(dist, torch.distributions.Distribution) for dist
                   in dists)
        return Box(auxiliary=dists[0], switch=dists[1])

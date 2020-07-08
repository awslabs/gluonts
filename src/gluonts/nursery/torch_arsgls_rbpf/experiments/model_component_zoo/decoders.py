import numpy as np
from torch import nn

from torch_extensions.layers_with_init import Conv2d, Linear
from torch_extensions.distributions.conditional_parametrised_distribution import (
    ParametrisedConditionalDistribution,
)
from utils.utils import (
    compute_cnn_output_filters_and_dims,
    Reshape,
    IndependentBernoulli,
)
from torch.distributions import MultivariateNormal
from utils.utils import Lambda
from torch_extensions.ops import batch_diag_matrix
from torch_extensions.mlp import MLP


class AuxiliaryToObsDecoderConvBernoulli(ParametrisedConditionalDistribution):
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
                Linear(
                    in_features=config.dims.auxiliary,
                    out_features=int(np.prod(shp_enc_out)),
                ),
                Reshape(shp_enc_out),  # TxPxB will be flattened before.
                Conv2d(
                    in_channels=shp_enc_out[0],
                    out_channels=config.dims_filter[-1]
                    * config.upscale_factor ** 2,
                    kernel_size=config.kernel_sizes[-1],
                    stride=1,  # Pixelshuffle instead.
                    padding=config.paddings[-1],
                ),
                nn.PixelShuffle(upscale_factor=config.upscale_factor),
                nn.ReLU(),
                Conv2d(
                    in_channels=config.dims_filter[-1],
                    out_channels=config.dims_filter[-2]
                    * config.upscale_factor ** 2,
                    kernel_size=config.kernel_sizes[-2],
                    stride=1,  # Pixelshuffle instead.
                    padding=config.paddings[-2],
                ),
                nn.PixelShuffle(upscale_factor=config.upscale_factor),
                nn.ReLU(),
                Conv2d(
                    in_channels=config.dims_filter[-2],
                    out_channels=config.dims_filter[-3]
                    * config.upscale_factor ** 2,
                    kernel_size=config.kernel_sizes[-3],
                    stride=1,  # Pixelshuffle instead.
                    padding=config.paddings[-3],
                ),
                nn.PixelShuffle(upscale_factor=config.upscale_factor),
                nn.ReLU(),
            ),
            dist_params=nn.ModuleDict(
                {
                    "logits": nn.Sequential(
                        Conv2d(
                            in_channels=config.dims_filter[-3],
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                        Reshape((config.dims.target,)),
                    )
                }
            ),
            dist_cls=IndependentBernoulli,
        )


class AuxiliaryToObsDecoderMlpGaussian(ParametrisedConditionalDistribution):
    def __init__(self, config):
        dim_in = config.dims.auxiliary
        dim_out = config.dims.target
        dims_stem = config.dims_decoder
        activations_stem = config.activations_decoder
        dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in

        super().__init__(
            stem=MLP(
                dim_in=dim_in,
                dims=dims_stem,
                activations=activations_stem,
            ),
            dist_params=nn.ModuleDict(
                {
                    "loc": nn.Sequential(
                        Linear(
                            in_features=dim_in_dist_params,
                            out_features=dim_out,
                        ),
                    ),
                    "scale_tril": nn.Sequential(
                        Linear(dim_in_dist_params, dim_out),
                        Lambda(fn=lambda x: x - 2),
                        # start with smaller scale to reduce noise early.
                        nn.Softplus(),
                        Lambda(fn=lambda x: x + 1e-6),
                        Lambda(fn=batch_diag_matrix),
                    ),
                }
            ),
            dist_cls=MultivariateNormal,
        )

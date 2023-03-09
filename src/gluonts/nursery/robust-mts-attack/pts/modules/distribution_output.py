from abc import ABC, abstractclassmethod
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Beta,
    NegativeBinomial,
    StudentT,
    Normal,
    Categorical,
    MixtureSameFamily,
    Independent,
    LowRankMultivariateNormal,
    MultivariateNormal,
    TransformedDistribution,
    AffineTransform,
    Poisson,
)

from pts.distributions import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomial,
    PiecewiseLinear,
    TransformedPiecewiseLinear,
    ImplicitQuantile,
    TransformedImplicitQuantile,
)
from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    LambdaLayer,
    PtArgProj,
)
from pts.modules.iqn_modules import ImplicitQuantileModule


class IndependentDistributionOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim

    @property
    def event_shape(self) -> Tuple:
        if self.dim is None:
            return ()
        else:
            return (self.dim,)

    def independent(self, distr: Distribution) -> Distribution:
        if self.dim is None:
            return distr

        return Independent(distr, 1)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:

        distr = self.independent(self.distr_cls(*distr_args))
        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])


class NormalOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls: type = Normal

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, loc, scale):
        scale = F.softplus(scale)
        return loc.squeeze(-1), scale.squeeze(-1)


class IndependentNormalOutput(NormalOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        warnings.warn(
            "IndependentNormalOutput is deprecated. Use NormalOutput instead.",
            DeprecationWarning,
        )


class BetaOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"concentration1": 1, "concentration0": 1}
    distr_cls: type = Beta

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, concentration1, concentration0):
        concentration1 = F.softplus(concentration1) + 1e-8
        concentration0 = F.softplus(concentration0) + 1e-8
        return concentration1.squeeze(-1), concentration0.squeeze(-1)


class PoissonOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1}
    distr_cls: type = Poisson

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, rate):
        rate_pos = F.softplus(rate).clone()

        return (rate_pos.squeeze(-1),)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        (rate,) = distr_args

        if scale is not None:
            rate *= scale

        return self.independent(Poisson(rate))


class ZeroInflatedPoissonOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"gate": 1, "rate": 1}
    distr_cls: type = ZeroInflatedPoisson

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, gate, rate):
        gate_unit = torch.sigmoid(gate).clone()
        rate_pos = F.softplus(rate).clone()

        return gate_unit.squeeze(-1), rate_pos.squeeze(-1)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        gate, rate = distr_args

        if scale is not None:
            rate *= scale

        return self.independent(ZeroInflatedPoisson(gate=gate, rate=rate))


class NegativeBinomialOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    distr_cls: type = NegativeBinomial

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, total_count, logits):
        total_count = F.softplus(total_count)
        return total_count.squeeze(-1), logits.squeeze(-1)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            logits += scale.log()

        return self.independent(
            NegativeBinomial(total_count=total_count, logits=logits)
        )


class ZeroInflatedNegativeBinomialOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"gate": 1, "total_count": 1, "logits": 1}
    distr_cls: type = ZeroInflatedNegativeBinomial

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, gate, total_count, logits):
        gate = torch.sigmoid(gate)
        total_count = F.softplus(total_count)
        return gate.squeeze(-1), total_count.squeeze(-1), logits.squeeze(-1)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        gate, total_count, logits = distr_args

        if scale is not None:
            logits += scale.log()

        return self.independent(
            ZeroInflatedNegativeBinomial(
                gate=gate, total_count=total_count, logits=logits
            )
        )


class StudentTOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distr_cls: type = StudentT

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, df, loc, scale):
        scale = F.softplus(scale)
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


class StudentTMixtureOutput(DistributionOutput):
    @validated()
    def __init__(self, components: int = 1) -> None:
        self.components = components
        self.args_dim = {
            "mix_logits": components,
            "df": components,
            "loc": components,
            "scale": components,
        }

    @classmethod
    def domain_map(cls, mix_logits, df, loc, scale):
        scale = F.softplus(scale)
        df = 2.0 + F.softplus(df)
        return (
            mix_logits.squeeze(-1),
            df.squeeze(-1),
            loc.squeeze(-1),
            scale.squeeze(-1),
        )

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        mix_logits, df, loc, dist_scale = distr_args

        distr = MixtureSameFamily(
            Categorical(logits=mix_logits), StudentT(df, loc, dist_scale)
        )
        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])

    @property
    def event_shape(self) -> Tuple:
        return ()


class PiecewiseLinearOutput(DistributionOutput):
    distr_cls: type = PiecewiseLinear

    @validated()
    def __init__(self, num_pieces: int) -> None:
        super().__init__(self)
        assert (
            isinstance(num_pieces, int) and num_pieces > 1
        ), "num_pieces should be an integer larger than 1"

        self.num_pieces = num_pieces
        self.args_dim = {"gamma": 1, "slopes": num_pieces, "knot_spacings": num_pieces}

    @classmethod
    def domain_map(cls, gamma, slopes, knot_spacings):
        # slopes of the pieces are non-negative
        slopes_proj = F.softplus(slopes) + 1e-4

        # the spacing between the knots should be in [0, 1] and sum to 1
        knot_spacings_proj = torch.softmax(knot_spacings, dim=-1)

        return gamma.squeeze(axis=-1), slopes_proj, knot_spacings_proj

    def distribution(
        self,
        distr_args,
        scale: Optional[torch.Tensor] = None,
    ) -> PiecewiseLinear:
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedPiecewiseLinear(
                distr, [AffineTransform(loc=0, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


class NormalMixtureOutput(DistributionOutput):
    @validated()
    def __init__(self, components: int = 1) -> None:
        self.components = components
        self.args_dim = {
            "mix_logits": components,
            "loc": components,
            "scale": components,
        }

    @classmethod
    def domain_map(cls, mix_logits, loc, scale):
        scale = F.softplus(scale)
        return mix_logits.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        mix_logits, loc, dist_scale = distr_args

        distr = MixtureSameFamily(
            Categorical(logits=mix_logits), Normal(loc, dist_scale)
        )
        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])

    @property
    def event_shape(self) -> Tuple:
        return ()


class LowRankMultivariateNormalOutput(DistributionOutput):
    @validated()
    def __init__(
        self,
        dim: int,
        rank: int,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
    ) -> None:
        self.distr_cls = LowRankMultivariateNormal
        self.dim = dim
        self.rank = rank
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum
        self.args_dim = {"loc": dim, "cov_factor": dim * rank, "cov_diag": dim}

    def domain_map(self, loc, cov_factor, cov_diag):
        diag_bias = (
            self.inv_softplus(self.sigma_init ** 2) if self.sigma_init > 0.0 else 0.0
        )

        shape = cov_factor.shape[:-1] + (self.dim, self.rank)
        cov_factor = cov_factor.reshape(shape)
        cov_diag = F.softplus(cov_diag + diag_bias) + self.sigma_minimum ** 2

        return loc, cov_factor, cov_diag

    def inv_softplus(self, y):
        if y < 20.0:
            return np.log(np.exp(y) - 1.0)
        else:
            return y

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class MultivariateNormalOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        self.args_dim = {"loc": dim, "scale_tril": dim * dim}
        self.dim = dim

    def domain_map(self, loc, scale):
        d = self.dim
        device = scale.device

        shape = scale.shape[:-1] + (d, d)
        scale = scale.reshape(shape)

        scale_diag = F.softplus(scale * torch.eye(d, device=device)) * torch.eye(
            d, device=device
        )

        mask = torch.tril(torch.ones_like(scale), diagonal=-1)
        scale_tril = (scale * mask) + scale_diag

        return loc, scale_tril

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        loc, scale_tri = distr_args
        distr = MultivariateNormal(loc=loc, scale_tril=scale_tri)

        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class FlowOutput(DistributionOutput):
    @validated()
    def __init__(self, flow, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.flow = flow
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.flow.scale = scale
        self.flow.cond = cond

        return self.flow

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class DiffusionOutput(DistributionOutput):
    @validated()
    def __init__(self, diffusion, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.diffusion = diffusion
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.diffusion.scale = scale
        self.diffusion.cond = cond

        return self.diffusion

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class QuantilePtArgProj(PtArgProj):
    def __init__(
        self,
        in_features: int,
        output_domain_cls: nn.Module,
        args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[torch.Tensor]],
        **kwargs,
    ):
        super().__init__(in_features, args_dim, domain_map, **kwargs)
        self.output_domain_cls = output_domain_cls
        self.proj = ImplicitQuantileModule(in_features, output_domain_cls)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        forecast_length = x.shape[1]
        device = x.device
        taus = torch.rand(size=(batch_size, forecast_length), device=device)
        self.register_buffer("taus", taus)
        self.register_buffer("nn_ouput", x.clone().detach())
        predicted_quantiles = self.proj(x, taus)
        return self.domain_map(predicted_quantiles)


class ImplicitQuantileOutput(IndependentDistributionOutput):
    distr_cls: type = ImplicitQuantile
    in_features = 1
    args_dim = {"quantile_function": 1}
    output_domain_cls: type = nn.Module
    quantile_arg_proj: type = nn.Module

    @validated()
    def __init__(self, output_domain: str) -> None:
        super().__init__()
        self.set_output_domain_map(output_domain)
        self.set_args_proj()

    @classmethod
    def set_output_domain_map(cls, output_domain):
        available_domain_map_cls = {
            "Positive": nn.Softplus,
            "Real": nn.Identity,
            "Unit": nn.Softmax,
        }
        assert (
            output_domain in available_domain_map_cls.keys()
        ), "Only the following output domains are allowed: {}".format(
            available_domain_map_cls.keys()
        )
        output_domain_cls = available_domain_map_cls[output_domain]
        cls.output_domain_cls = output_domain_cls

    @classmethod
    def set_args_proj(cls):
        cls.quantile_arg_proj = QuantilePtArgProj(
            in_features=cls.in_features,
            output_domain_cls=cls.output_domain_cls,
            args_dim=cls.args_dim,
            domain_map=LambdaLayer(cls.domain_map),
        )

    @classmethod
    def domain_map(cls, *args):
        return args

    @classmethod
    def args_proj(cls, in_features):
        if in_features != cls.in_features:
            cls.in_features = in_features
            cls.set_args_proj()
        return cls.quantile_arg_proj

    def get_args_proj(self, in_features: int, prefix: Optional[str] = None):
        return self.args_proj(in_features)

    def distribution(
        self,
        distr_args,
        scale: Optional[torch.Tensor] = None,
    ) -> ImplicitQuantile:

        args_proj = self.get_args_proj(self.in_features)
        implicit_quantile_function = args_proj.proj.eval()
        distr = self.distr_cls(
            implicit_quantile_function=implicit_quantile_function,
            taus=list(args_proj.buffers())[0],
            nn_output=list(args_proj.buffers())[1],
            predicted_quantiles=distr_args,
        )
        if scale is None:
            return distr
        else:
            return TransformedImplicitQuantile(
                distr, [AffineTransform(loc=0, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()

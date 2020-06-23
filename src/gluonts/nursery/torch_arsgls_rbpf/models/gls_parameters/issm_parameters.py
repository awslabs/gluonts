import torch
from box import Box
from torch import nn
from models.gls_parameters.issm import ISSM, LevelISSM, LevelTrendISSM, \
    SeasonalityISSM, \
    CompositeISSM
from torch_extensions.ops import batch_diag_matrix, matvec
from models.gls_parameters.gls_parameters import GLSParameters


class ISSMParameters(GLSParameters):
    def __init__(self, issm: ISSM, n_base_R,
                 init_scale_R_diag: (float, tuple, list, None) = None,
                 **kwargs):
        # kwargs["init_scale_B"]=0.1,  # some small value - these are just biases.
        # kwargs["init_scale_D"]=0.1,
        kwargs["n_base_A"] = None
        kwargs["n_base_C"] = None
        assert kwargs["n_obs"] == 1
        super().__init__(n_base_R=n_base_R, init_scale_R_diag=init_scale_R_diag,
                         **kwargs)
        self.issm = issm
        # ***** Overwrite LRinv_logdiag *****
        # Note that the 1-4 (season 1, season 2, level, trend) R parameters are
        # shared per base-matrix. In most cases, where the link is shared,
        # we share the weights for all base parameters though anyways.
        if isinstance(self.issm, (LevelISSM, LevelTrendISSM, SeasonalityISSM)):
            n_params_R = 1
        elif isinstance(self.issm, CompositeISSM):
            if hasattr(self.issm, "nonseasonal_issm"):
                if isinstance(self.issm.nonseasonal_issm, LevelTrendISSM):
                    n_params_R = len(self.issm.seasonal_issms) + 2
                elif isinstance(self.issm.nonseasonal_issm, LevelISSM):
                    n_params_R = len(self.issm.seasonal_issms) + 1
                else:
                    raise Exception(
                        f"unknown nonseasonal_issm: {self.issm.nonseasonal_issm}")
            else:
                n_params_R = len(self.issm.seasonal_issms)
        else:
            raise NotImplementedError()

        self.LRinv_logdiag = nn.Parameter(
            self.make_cov_init(
                init_scale_cov_diag=init_scale_R_diag,
                n_base=n_base_R,
                dim_cov=n_params_R
            ),
            requires_grad=self.LRinv_logdiag.requires_grad,
        )

    def forward(self, switch, u_state, u_obs, seasonal_indicators) -> Box:
        weights = self.link_transformers(switch=switch)

        # biases
        B = torch.einsum("...k,koi->...oi", weights.B,
                         self.B) if self.B is not None else None
        D = torch.einsum("...k,koi->...oi", weights.D,
                         self.D) if self.D is not None else None
        b = self.compute_bias(s=switch, u=u_state, bias_fn=self.b_fn,
                              bias_matrix=B)
        d = self.compute_bias(s=switch, u=u_obs, bias_fn=self.d_fn,
                              bias_matrix=D)

        # transition and emission from ISSM
        _, C, R_diag_projector = self.issm(
            seasonal_indicators=seasonal_indicators)
        A = None  # instead of identity, we use None to avoid unnecessary computation

        # covariance matrices
        if self.make_cov_from_cholesky_avg:
            Q_diag = self.var_from_average_scales(
                weights=weights.Q,
                Linv_logdiag=self.LQinv_logdiag_limiter(self.LQinv_logdiag),
            )
            R_diag = self.var_from_average_scales(
                weights=weights.R,
                Linv_logdiag=self.LRinv_logdiag_limiter(self.LRinv_logdiag),
            )
        else:
            Q_diag = self.var_from_average_variances(
                weights=weights.Q,
                Linv_logdiag=self.LQinv_logdiag_limiter(self.LQinv_logdiag),
            )
            R_diag = self.var_from_average_variances(
                weights=weights.R,
                Linv_logdiag=self.LRinv_logdiag_limiter(self.LRinv_logdiag),
            )
        Q = batch_diag_matrix(Q_diag)
        R = batch_diag_matrix(matvec(R_diag_projector, R_diag))
        return Box(A=A, b=b, C=C, d=d, Q=Q, R=R)

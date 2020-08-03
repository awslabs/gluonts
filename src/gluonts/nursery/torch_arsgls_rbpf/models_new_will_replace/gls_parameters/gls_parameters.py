from typing import Optional, Union, Sequence
import math
import numpy as np
import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from models.gls_parameters.switch_link_functions import (
    IndividualLink,
    SharedLink,
    IdentityLink,
)
from torch_extensions.ops import (
    cov_and_chol_from_invcholesky_param,
    matvec,
    batch_diag_matrix,
    matmul,
)
from experiments.base_config import SwitchLinkType
from models_new_will_replace.base_gls import ControlInputs, GLSParams


def filter_out_none(dic: dict):
    return {key: val for key, val in dic.items() if val is not None}


class GLSParameters(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_ctrl_state: Optional[int],
        n_ctrl_obs: Optional[int],
        n_switch: int,
        n_base_A: Optional[int],
        n_base_B: Optional[int],
        n_base_C: Optional[int],
        n_base_D: Optional[int],
        n_base_R: int,
        n_base_Q: int,
        switch_link_type: SwitchLinkType,
        switch_link_dims_hidden: tuple = tuple(),
        switch_link_activations: nn.Module = nn.ReLU(),
        make_cov_from_cholesky_avg=False,
        b_fn: Optional[nn.Module] = None,
        d_fn: Optional[nn.Module] = None,
        init_scale_A: (float, None) = None,
        init_scale_B: (float, None) = None,
        init_scale_C: Optional[float] = None,
        init_scale_D: Optional[float] = None,
        init_scale_Q_diag: Optional[Union[float, Sequence[float]]] = None,
        init_scale_R_diag: Optional[Union[float, Sequence[float]]] = None,
        requires_grad_A: bool = True,
        requires_grad_B: bool = True,
        requires_grad_C: bool = True,
        requires_grad_D: bool = True,
        requires_grad_R: bool = True,
        requires_grad_Q: bool = True,
        full_cov_R: bool = True,
        full_cov_Q: bool = True,
        LQinv_logdiag_limiter: Optional[nn.Module] = None,
        LRinv_logdiag_limiter: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.make_cov_from_cholesky_avg = make_cov_from_cholesky_avg
        self.LQinv_logdiag_limiter = (
            LQinv_logdiag_limiter
            if LQinv_logdiag_limiter is not None
            else torch.nn.Identity()
        )
        self.LRinv_logdiag_limiter = (
            LRinv_logdiag_limiter
            if LRinv_logdiag_limiter is not None
            else torch.nn.Identity()
        )
        self.b_fn = b_fn
        self.d_fn = d_fn

        # ***** Switch Link function *****
        n_bases = [
            n
            for n in [
                n_base_A,
                n_base_B,
                n_base_C,
                n_base_D,
                n_base_R,
                n_base_Q,
            ]
            if n is not None
        ]
        if switch_link_type == SwitchLinkType.identity:
            assert (
                len(set(n_bases)) == 1 and n_bases[0] == n_switch
            ), f"n_base: {n_bases} should match switch dim {n_switch} " \
               f"when using identity link."
        elif switch_link_type == SwitchLinkType.shared:
            assert len(set(n_bases)) == 1

        names_and_dims_out = filter_out_none(
            {
                "A": n_base_A,
                "B": n_base_B,
                "C": n_base_C,
                "D": n_base_D,
                "R": n_base_R,
                "Q": n_base_Q,
            }
        )
        if switch_link_type.value == SwitchLinkType.individual.value:
            self.link_transformers = IndividualLink(
                dim_in=n_switch,
                names_and_dims_out=names_and_dims_out,
                dims_hidden=switch_link_dims_hidden,
                activations_hidden=switch_link_activations,
            )
        elif switch_link_type.value == SwitchLinkType.identity.value:
            self.link_transformers = IdentityLink(
                names=tuple(names_and_dims_out.keys())
            )
        elif switch_link_type.value == SwitchLinkType.shared.value:
            dims = [dim for dim in names_and_dims_out.values()]  # strip None
            assert len(set(dims)) == 1
            dim_out = dims[0]
            self.link_transformers = SharedLink(
                dim_in=n_switch,
                dim_out=dim_out,
                names=tuple(names_and_dims_out.keys()),
            )
        else:
            raise Exception(f"unknown switch link type: {switch_link_type}")

        # ***** Initialise GLS Parameters *****
        if n_base_A is not None:
            init_scale_A = init_scale_A if init_scale_A is not None else 1.0
            self.A = nn.Parameter(
                init_scale_A * torch.eye(n_state).repeat(n_base_A, 1, 1),
                requires_grad=requires_grad_A,
            )
            # self.A = nn.Parameter(
            #     init_scale_A * torch.nn.init.orthogonal_(torch.empty(n_base_A, n_state, n_state)),
            #     requires_grad=True,
            # )
        else:
            self.register_parameter("A", None)

        if n_base_B is not None:
            if init_scale_B is not None:
                self.B = nn.Parameter(
                    init_scale_B
                    * torch.randn(n_base_B, n_state, n_ctrl_state),
                    requires_grad=requires_grad_B,
                )
            else:
                self.B = nn.Parameter(
                    torch.stack(
                        [
                            kaiming_normal_(
                                tensor=torch.empty(n_state, n_ctrl_state),
                                nonlinearity="linear",
                            )
                            for n in range(n_base_B)
                        ],
                        dim=0,
                    ),
                    requires_grad=requires_grad_B,
                )
        else:
            self.register_parameter("B", None)

        if n_base_C is not None:
            if init_scale_C is not None:
                self.C = nn.Parameter(
                    init_scale_C * torch.randn(n_base_C, n_obs, n_state),
                    requires_grad=requires_grad_C,
                )
            else:
                self.C = nn.Parameter(
                    torch.stack(
                        [
                            kaiming_normal_(
                                tensor=torch.empty(n_obs, n_state),
                                nonlinearity="linear",
                            )
                            for n in range(n_base_C)
                        ],
                        dim=0,
                    ),
                    requires_grad=requires_grad_C,
                )
        else:
            self.register_parameter("C", None)

        if n_base_D is not None:
            if init_scale_D is not None:
                self.D = nn.Parameter(
                    init_scale_D * torch.randn(n_base_D, n_obs, n_ctrl_obs),
                    requires_grad=requires_grad_D,
                )
            else:
                self.D = nn.Parameter(
                    torch.stack(
                        [
                            kaiming_normal_(
                                tensor=torch.empty(n_obs, n_ctrl_obs),
                                nonlinearity="linear",
                            )
                            for n in range(n_base_D)
                        ],
                        dim=0,
                    ),
                    requires_grad=requires_grad_D,
                )
        else:
            self.register_parameter("D", None)

        if n_base_Q is not None:
            if full_cov_Q:  # tril part is always initialised zero
                self.LQinv_tril = nn.Parameter(
                    torch.zeros((n_base_Q, n_obs, n_obs)),
                    requires_grad=requires_grad_Q,
                )
            else:
                self.register_parameter("LQinv_tril", None)
            init_scale_Q_diag = (
                init_scale_Q_diag
                if init_scale_Q_diag is not None
                else [1e-4, 1e0]
            )
            if isinstance(init_scale_Q_diag, (list, tuple)):
                self.LQinv_logdiag = nn.Parameter(
                    self.make_cov_init(
                        init_scale_cov_diag=init_scale_Q_diag,
                        n_base=n_base_Q,
                        dim_cov=n_obs,
                    ),
                    requires_grad=requires_grad_Q,
                )
            else:
                self.LQinv_logdiag = nn.Parameter(
                    torch.ones((n_base_Q, n_obs))
                    * -math.log(init_scale_Q_diag),
                    requires_grad=requires_grad_Q,
                )

        if n_base_R is not None:
            if full_cov_R:  # tril part is always initialised zero
                self.LRinv_tril = nn.Parameter(
                    torch.zeros((n_base_R, n_state, n_state)),
                    requires_grad=requires_grad_R,
                )
            else:
                self.register_parameter("LRinv_tril", None)
            init_scale_R_diag = (
                init_scale_R_diag
                if init_scale_R_diag is not None
                else [1e-4, 1e0]
            )
            if isinstance(init_scale_R_diag, (list, tuple)):
                self.LRinv_logdiag = nn.Parameter(
                    self.make_cov_init(
                        init_scale_cov_diag=init_scale_R_diag,
                        n_base=n_base_R,
                        dim_cov=n_state,
                    ),
                    requires_grad=requires_grad_R,
                )
            else:
                self.LRinv_logdiag = nn.Parameter(
                    torch.ones((n_base_R, n_state))
                    * -math.log(init_scale_R_diag),
                    requires_grad=requires_grad_R,
                )

    @staticmethod
    def make_cov_init(init_scale_cov_diag: (tuple, list), n_base, dim_cov):
        def _make_single_cov_linspace_init(
            init_scale_cov_diag: (tuple, list), n_base
        ):
            assert len(init_scale_cov_diag) == 2
            log_scale_cov = torch.log(torch.tensor(init_scale_cov_diag))
            Lmatinv_logdiag = -torch.linspace(
                log_scale_cov[0], log_scale_cov[1], n_base
            )
            idxs = list(range(Lmatinv_logdiag.shape[-1]))
            np.random.shuffle(idxs)
            Lmatinv_logdiag = Lmatinv_logdiag[..., torch.tensor(idxs)]
            return Lmatinv_logdiag

        cov = _make_single_cov_linspace_init(
            init_scale_cov_diag=init_scale_cov_diag, n_base=n_base,
        )
        cov = cov[..., None].repeat(
            cov.ndim * (1,) + (dim_cov,)
        )  # same scale for all dims.
        return cov

    @staticmethod
    def var_from_average_scales(
        weights: torch.Tensor, Linv_logdiag: torch.Tensor
    ):
        Lmat_diag = torch.exp(-1 * Linv_logdiag)
        Lmat_diag_weighted = torch.einsum("...k,kq->...q", weights, Lmat_diag)
        mat_diag_weighted = Lmat_diag_weighted ** 2

        return mat_diag_weighted, Lmat_diag_weighted

    @staticmethod
    def var_from_average_variances(
        weights: torch.Tensor, Linv_logdiag: torch.Tensor
    ):
        mat_diag = torch.exp(-2 * Linv_logdiag)
        mat_diag_weighted = torch.einsum("...k,kq->...q", weights, mat_diag)
        Lmat_diag_weighted = torch.sqrt(mat_diag_weighted)
        return mat_diag_weighted, Lmat_diag_weighted

    @staticmethod
    def var_from_average_log_scales(
        weights: torch.Tensor, Linv_logdiag: torch.Tensor
    ):
        Linv_logdiag_weighted = torch.einsum(
            "...k,kq->...q", weights, Linv_logdiag
        )
        mat_diag_weighted = torch.exp(-2 * Linv_logdiag_weighted)
        Lmat_diag_weighted = torch.exp(-1 * Linv_logdiag_weighted)
        return mat_diag_weighted, Lmat_diag_weighted

    @staticmethod
    def cov_from_average_scales(
        weights: torch.Tensor,
        Linv_logdiag: torch.Tensor,
        Linv_tril: (torch.Tensor, None),
    ):
        if Linv_tril is None:
            mat_diag_weighted, Lmat_diag_weighted = GLSParameters\
                .var_from_average_scales(
                    weights=weights, Linv_logdiag=Linv_logdiag,
            )
            mat_weighted = batch_diag_matrix(mat_diag_weighted)
            Lmat_weighted = batch_diag_matrix(Lmat_diag_weighted)
        else:
            Lmat = torch.inverse(
                torch.tril(Linv_tril, -1)
                + batch_diag_matrix(torch.exp(Linv_logdiag))
            )
            Lmat_weighted = torch.einsum("...k,koi->...oi", weights, Lmat)
            mat_weighted = matmul(
                Lmat_weighted, Lmat_weighted.transpose(-1, -2)
            )  # LL^T
        return mat_weighted, Lmat_weighted

    @staticmethod
    def cov_from_average_variances(
        weights: torch.Tensor,
        Linv_logdiag: torch.Tensor,
        Linv_tril: (torch.Tensor, None),
    ):
        if Linv_tril is None:
            mat_diag_weighted, Lmat_diag_weighted = GLSParameters\
                .var_from_average_variances(
                    weights=weights, Linv_logdiag=Linv_logdiag,
            )
            mat_weighted = batch_diag_matrix(mat_diag_weighted)
            Lmat_weighted = batch_diag_matrix(Lmat_diag_weighted)
        else:
            mat, _ = cov_and_chol_from_invcholesky_param(
                Linv_tril=Linv_tril, Linv_logdiag=Linv_logdiag,
            )
            mat_weighted = torch.einsum("...k,kq->...q", weights, mat)
            Lmat_weighted = torch.cholesky(mat_weighted)
        return mat_weighted, Lmat_weighted

    @staticmethod
    def cov_from_average_log_scales(
        weights: torch.Tensor,
        Linv_logdiag: torch.Tensor,
        Linv_tril: (torch.Tensor, None),
    ):
        if Linv_tril is None:
            mat_diag_weighted, Lmat_diag_weighted = GLSParameters\
                .var_from_average_log_scales(
                    weights=weights, Linv_logdiag=Linv_logdiag,
            )
            mat_weighted = batch_diag_matrix(mat_diag_weighted)
            Lmat_weighted = batch_diag_matrix(Lmat_diag_weighted)
        else:
            raise Exception("No can do.")
        return mat_weighted, Lmat_weighted

    @staticmethod
    def compute_bias(s, u=None, bias_fn=None, bias_matrix=None):
        if bias_fn is None and bias_matrix is None:
            b = None
        else:
            b_nonlin = bias_fn(s) if bias_fn is not None else 0.0
            b_lin = matvec(bias_matrix, u) if bias_matrix is not None else 0.0
            b = b_lin + b_nonlin
        return b

    def forward(self, switch, controls: Optional[ControlInputs]) -> GLSParams:
        weights = self.link_transformers(switch)

        # biases to state (B/b) and observation (D/d)
        B = (
            torch.einsum("...k,koi->...oi", weights.B, self.B)
            if self.B is not None
            else None
        )
        D = (
            torch.einsum("...k,koi->...oi", weights.D, self.D)
            if self.D is not None
            else None
        )
        b = self.compute_bias(
            s=switch,
            u=controls.state if controls is not None else None,
            bias_fn=self.b_fn,
            bias_matrix=B,
        )
        d = self.compute_bias(
            s=switch,
            u=controls.target if controls is not None else None,
            bias_fn=self.d_fn,
            bias_matrix=D,
        )

        # transition (A) and emission (C)
        A = torch.einsum("...k,koi->...oi", weights.A, self.A)
        C = torch.einsum("...k,koi->...oi", weights.C, self.C)

        # covariances matrices transition (R) and emission (Q)
        if self.make_cov_from_cholesky_avg:
            Q, LQ = self.cov_from_average_scales(
                weights=weights.Q,
                Linv_logdiag=self.LQinv_logdiag_limiter(self.LQinv_logdiag),
                Linv_tril=self.LQinv_tril,
            )
            R, LR = self.cov_from_average_scales(
                weights=weights.R,
                Linv_logdiag=self.LRinv_logdiag_limiter(self.LRinv_logdiag),
                Linv_tril=self.LRinv_tril,
            )
        else:
            Q, LQ = self.cov_from_average_variances(
                weights=weights.Q,
                Linv_logdiag=self.LQinv_logdiag_limiter(self.LQinv_logdiag),
                Linv_tril=self.LQinv_tril,
            )
            R, LR = self.cov_from_average_variances(
                weights=weights.R,
                Linv_logdiag=self.LRinv_logdiag_limiter(self.LRinv_logdiag),
                Linv_tril=self.LRinv_tril,
            )
        # return B and D because the loss and smoothing functions use B / D atm
        # and do not support b / d (although that is straightforward to do).
        return GLSParams(
            A=A, b=b, C=C, d=d, Q=Q, R=R, B=B, D=D, LQ=LQ, LR=LR,
        )


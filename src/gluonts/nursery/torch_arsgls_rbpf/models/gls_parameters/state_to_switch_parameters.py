from typing import Optional
import math
from box import Box
import numpy as np
import torch
from torch import nn
from experiments.base_config import SwitchLinkType
from models.gls_parameters.switch_link_functions import (
    IdentityLink,
    IndividualLink,
    SharedLink,
)
from models.gls_parameters.gls_parameters import GLSParameters


class StateToSwitchParams(nn.Module):
    def __init__(
        self,
        n_switch,
        n_state,
        n_base_F,
        n_base_S,
        init_scale_S_diag=(1e-4, 1e0,),
        requires_grad_S: bool = True,
        full_cov_S: bool = False,
        make_cov_from_cholesky_avg: bool = True,
        switch_link: (nn.Module, None) = None,
        switch_link_type: (SwitchLinkType, None) = None,
        LSinv_logdiag_scaling: Optional[float] = None,
        F_scaling: Optional[float] = None,
    ):
        super().__init__()
        assert len({switch_link is None, switch_link_type is None}) == 2
        self.make_cov_from_cholesky_avg = make_cov_from_cholesky_avg
        self._LSinv_logdiag_scaling = LSinv_logdiag_scaling
        self._F_scaling = F_scaling

        if switch_link is not None:
            self.link_transformers = switch_link
        elif switch_link_type is not None:
            if switch_link_type.value == SwitchLinkType.individual.value:
                self.link_transformers = IndividualLink(
                    dim_in=n_switch,
                    names_and_dims={"S": n_base_S, "F": n_base_F,},
                )
            elif switch_link_type.value == SwitchLinkType.identity.value:
                names = ("S", "F")
                self.link_transformers = IdentityLink(names=names)
            elif switch_link_type.value == SwitchLinkType.shared.value:
                dims = [val for val in {n_base_S, n_base_F} if val is not None]
                assert len(set(dims)) == 1
                dim_out = dims[0]
                self.link_transformers = SharedLink(
                    dim_in=n_switch, dim_out=dim_out, names=("S", "F")
                )
            else:
                raise Exception(
                    f"unknown switch link type: {switch_link_type}"
                )
        else:
            raise Exception()

        self._F = nn.Parameter(
            torch.nn.init.orthogonal_(
                torch.empty(n_base_F, n_switch, n_state)
            ),
            requires_grad=True,
        )
        self._F.data /= self._F_scaling

        if full_cov_S:  # tril part is always initialised zero
            self.LSinv_tril = nn.Parameter(
                torch.zeros((n_base_S, n_switch, n_switch)),
                requires_grad=requires_grad_S,
            )
        else:
            self.register_parameter("LSinv_tril", None)

        if isinstance(init_scale_S_diag, (list, tuple)):
            assert len(init_scale_S_diag) == 2

            def make_lower_logdiag(init_scale, n_base):
                log_scale = torch.log(torch.tensor(init_scale))
                Linv_logdiag = -torch.linspace(
                    log_scale[0], log_scale[1], n_base
                )
                idxs = list(range(Linv_logdiag.shape[-1]))
                np.random.shuffle(idxs)
                Linv_logdiag = Linv_logdiag[..., torch.tensor(idxs)]
                return Linv_logdiag

            self._LSinv_logdiag = nn.Parameter(
                torch.stack(
                    [
                        make_lower_logdiag(
                            init_scale=init_scale_S_diag, n_base=n_base_S
                        )
                        for _ in range(n_switch)
                    ],
                    dim=-1,
                ),
                # LSinv_logdiag[:, None].repeat(1, n_switch),
                requires_grad=requires_grad_S,
            )
        else:
            self._LSinv_logdiag = nn.Parameter(
                torch.ones((n_base_S, n_switch))
                * -math.log(init_scale_S_diag),
                requires_grad=requires_grad_S,
            )
        self._LSinv_logdiag.data /= self._LSinv_logdiag_scaling

    @property
    def LSinv_logdiag(self):
        return GLSParameters._scale_mat(
            self._LSinv_logdiag, self._LSinv_logdiag_scaling,
        )

    @property
    def F(self):
        return GLSParameters._scale_mat(self._F, self._F_scaling)

    def forward(self, switch: torch.Tensor) -> Box:
        weights = self.link_transformers(switch=switch)
        F = torch.einsum("...k,koi->...oi", weights.F, self.F)

        if self.make_cov_from_cholesky_avg:
            S, LS = GLSParameters.cov_from_average_scales(
                weights=weights.S,
                Linv_logdiag=self.LSinv_logdiag,
                Linv_tril=self.LSinv_tril,
            )

        else:
            S, LS = GLSParameters.cov_from_average_variances(
                weights=weights.S,
                Linv_logdiag=self.LSinv_logdiag,
                Linv_tril=self.LSinv_tril,
            )

        return Box(F=F, S=S, LS=LS)

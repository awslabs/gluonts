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
from torch_extensions.ops import batch_diag_matrix, matmul


class StateToSwitchParams(nn.Module):
    def __init__(
        self,
        n_switch,
        n_state,
        n_base_F,
        n_base_S,
        init_scale_S_diag=(1e-4, 1e0,),
        trainable_S=True,
        switch_link: (nn.Module, None) = None,
        switch_link_type: (SwitchLinkType, None) = None,
    ):
        super().__init__()
        assert len({switch_link is None, switch_link_type is None}) == 2

        if switch_link is not None:
            self.link_transformers = switch_link
        elif switch_link_type is not None:
            if switch_link_type.value == SwitchLinkType.individual.value:
                self.link_transformers = IndividualLink(
                    dim_in=n_switch,
                    names_and_dims_out={"S": n_base_S, "F": n_base_F,},
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

        self.F = nn.Parameter(
            torch.nn.init.orthogonal_(
                torch.empty(n_base_F, n_switch, n_state)
            ),
            requires_grad=True,
        )
        # TODO: optionally learnable.
        self.LSinv_tril = nn.Parameter(
            torch.zeros((n_base_S, n_switch, n_switch)), requires_grad=False,
        )
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

            self.LSinv_logdiag = nn.Parameter(
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
                requires_grad=True if trainable_S else False,
            )
        else:
            self.LSinv_logdiag = nn.Parameter(
                torch.ones((n_base_S, n_switch))
                * -math.log(init_scale_S_diag),
                requires_grad=True if trainable_S else False,
            )

    def forward(self, switch):
        weights = self.link_transformers(switch=switch)
        F = torch.einsum("...k,koi->...oi", weights.F, self.F)
        LS_basemats = torch.inverse(
            torch.tril(self.LSinv_tril, -1)
            + batch_diag_matrix(torch.exp(self.LSinv_logdiag))
        )
        LS = torch.einsum("...k,koi->...oi", weights.S, LS_basemats)
        S = matmul(LS, LS.transpose(-1, -2))
        return Box(F=F, S=S)

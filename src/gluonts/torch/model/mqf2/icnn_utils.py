# LICENSE: EXTERNAL
# MIT License
#
# Copyright (c) 2021 Chin-Wei Huang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Tuple, List

from cpflows.flows import SequentialFlow, DeepConvexFlow

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


class DeepConvexNet(DeepConvexFlow):
    r"""
    Class that takes a partially input convex neural network (picnn)
    as input and equips it with functions of logdet
    computation (both estimation and exact computation)

    This class is based on DeepConvexFlow of the CP-Flow
    repo (https://github.com/CW-Huang/CP-Flow)

    For details of the logdet estimator, see
    ``Convex potential flows: Universal probability distributions
    with optimal transport and convex optimization``

    Parameters
    ----------
    picnn
        A partially input convex neural network (picnn)
    dim
        Dimension of the input
    is_energy_score
        Indicates if energy score is used as the objective function
        If yes, the network is not required to be strictly convex,
        so we can just use the picnn
        otherwise, a quadratic term is added to the output of picnn
        to render it strictly convex
    m1
        Dimension of the Krylov subspace of the Lanczos tridiagonalization
        used in approximating H of logdet(H)
    m2
        Iteration number of the conjugate gradient algorithm
        used to approximate logdet(H)
    rtol
        relative tolerance of the conjugate gradient algorithm
    atol
        absolute tolerance of the conjugate gradient algorithm
    """

    def __init__(
        self,
        picnn: torch.nn.Module,
        dim: int,
        is_energy_score: bool = False,
        estimate_logdet: bool = False,
        m1: int = 10,
        m2: Optional[int] = None,
        rtol: float = 0.0,
        atol: float = 1e-3,
    ) -> None:
        super().__init__(
            picnn,
            dim,
            m1=m1,
            m2=m2,
            rtol=rtol,
            atol=atol,
        )

        self.picnn = self.icnn
        self.is_energy_score = is_energy_score
        self.estimate_logdet = estimate_logdet

    def get_potential(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n = x.size(0)
        output = self.picnn(x, context)

        if self.is_energy_score:
            return output
        else:
            return (
                F.softplus(self.w1) * output
                + F.softplus(self.w0)
                * (x.view(n, -1) ** 2).sum(1, keepdim=True)
                / 2
            )

    def forward_transform(
        self,
        x: torch.Tensor,
        logdet: Optional[torch.Tensor] = 0,
        context: Optional[torch.Tensor] = None,
        extra: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.estimate_logdet:
            return self.forward_transform_stochastic(
                x, logdet, context=context, extra=extra
            )
        else:
            return self.forward_transform_bruteforce(
                x, logdet, context=context
            )


class SequentialNet(SequentialFlow):
    r"""
    Class that combines a list of DeepConvexNet and ActNorm
    layers and provides energy score computation

    This class is based on SequentialFlow of the CP-Flow repo
    (https://github.com/CW-Huang/CP-Flow)

    Parameters
    ----------
    networks
        list of DeepConvexNet and/or ActNorm instances
    """

    def __init__(self, networks: List[torch.nn.Module]) -> None:
        super().__init__(networks)
        self.networks = self.flows

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for network in self.networks:
            if isinstance(network, DeepConvexNet):
                x = network.forward(x, context=context)
            else:
                x = network.forward(x)
        return x

    def es_sample(
        self, hidden_state: torch.Tensor, dimension: int
    ) -> torch.Tensor:
        """
        Auxiliary function for energy score computation Drawing samples
        conditioned on the hidden state.

        Parameters
        ----------
        hidden_state
            hidden_state which the samples conditioned
            on (num_samples, hidden_size)
        dimension
            dimension of the input

        Returns
        -------
        samples
            samples drawn (num_samples, dimension)
        """

        num_samples = hidden_state.shape[0]

        zero = torch.tensor(
            0, dtype=hidden_state.dtype, device=hidden_state.device
        )
        one = torch.ones_like(zero)
        standard_normal = Normal(zero, one)

        samples = self.forward(
            standard_normal.sample([num_samples * dimension]).view(
                num_samples, dimension
            ),
            context=hidden_state,
        )

        return samples

    def energy_score(
        self,
        z: torch.Tensor,
        hidden_state: torch.Tensor,
        es_num_samples: int = 50,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes the (approximated) energy score sum_i ES(g,z_i), where
        ES(g,z_i) =

        -1/(2*es_num_samples^2) * sum_{w,w'} ||w-w'||_2^beta
        + 1/es_num_samples * sum_{w''} ||w''-z_i||_2^beta,
        w's are samples drawn from the
        quantile function g(., h_i) (gradient of picnn),
        h_i is the hidden state associated with z_i,
        and es_num_samples is the number of samples drawn
        for each of w, w', w'' in energy score approximation

        Parameters
        ----------
        z
            Observations (numel_batch, dimension)
        hidden_state
            Hidden state (numel_batch, hidden_size)
        es_num_samples
            Number of samples drawn for each of w, w', w''
            in energy score approximation
        beta
            Hyperparameter of the energy score, see the formula above
        Returns
        -------
        loss
            energy score (numel_batch)
        """

        numel_batch, dimension = z.shape[0], z.shape[1]

        hidden_state_repeat = hidden_state.repeat_interleave(
            repeats=es_num_samples, dim=0
        )

        w = self.es_sample(hidden_state_repeat, dimension)
        w_prime = self.es_sample(hidden_state_repeat, dimension)

        first_term = (
            torch.norm(
                w.view(numel_batch, 1, es_num_samples, dimension)
                - w_prime.view(numel_batch, es_num_samples, 1, dimension),
                dim=-1,
            )
            ** beta
        )

        mean_first_term = torch.mean(first_term.view(numel_batch, -1), dim=-1)

        # since both tensors are huge (numel_batch*es_num_samples, dimension),
        # delete to free up GPU memories
        del w, w_prime

        z_repeat = z.repeat_interleave(repeats=es_num_samples, dim=0)
        w_bar = self.es_sample(hidden_state_repeat, dimension)

        second_term = (
            torch.norm(
                w_bar.view(numel_batch, es_num_samples, dimension)
                - z_repeat.view(numel_batch, es_num_samples, dimension),
                dim=-1,
            )
            ** beta
        )

        mean_second_term = torch.mean(
            second_term.view(numel_batch, -1), dim=-1
        )

        loss = -0.5 * mean_first_term + mean_second_term

        return loss

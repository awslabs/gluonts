# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import AffineTransform, TransformedDistribution

from gluonts.core.component import validated

from .distribution_output import DistributionOutput


class ISQF(torch.distributions.Distribution):
    r"""
    Distribution class for the Incremental (Spline) Quantile Function in the
    paper ``Learning Quantile Functions without Quantile Crossing for
    Distribution-free Time Series Forecasting``
    by Park, Robinson, Aubet, Kan, Gasthaus, Wang
    Parameters
    ----------
    spline_knots, spline_heights
        Tensor parametrizing the x-positions (y-positions) of the spline knots
        Shape: (*batch_shape, (num_qk-1), num_pieces)
    qk_x, qk_y
        Tensor containing the increasing x-positions (y-positions) of the
        quantile knots, Shape: (*batch_shape, num_qk)
    beta_l, beta_r
        Tensor containing the non-negative learnable parameter of the
        left (right) tail, Shape: (*batch_shape,)
    """

    def __init__(
        self,
        spline_knots: torch.Tensor,
        spline_heights: torch.Tensor,
        beta_l: torch.Tensor,
        beta_r: torch.Tensor,
        qk_y: torch.Tensor,
        qk_x: torch.Tensor,
        tol: float = 1e-4,
        validate_args: bool = False,
    ) -> None:

        self.num_qk, self.num_pieces = qk_y.shape[-1], spline_knots.shape[-1]
        self.spline_knots, self.spline_heights = spline_knots, spline_heights
        self.beta_l, self.beta_r = beta_l, beta_r
        self.qk_y_all = qk_y
        self.tol = tol

        super().__init__(
            batch_shape=self.batch_shape, validate_args=validate_args
        )

        # Get quantile knots (qk) parameters
        (
            self.qk_x,
            self.qk_x_plus,
            self.qk_x_l,
            self.qk_x_r,
        ) = ISQF.parameterize_qk(qk_x)
        (
            self.qk_y,
            self.qk_y_plus,
            self.qk_y_l,
            self.qk_y_r,
        ) = ISQF.parameterize_qk(qk_y)

        # Get spline knots (sk) parameters
        self.sk_y, self.delta_sk_y = ISQF.parameterize_spline(
            self.spline_heights,
            self.qk_y,
            self.qk_y_plus,
            self.tol,
        )
        self.sk_x, self.delta_sk_x = ISQF.parameterize_spline(
            self.spline_knots,
            self.qk_x,
            self.qk_x_plus,
            self.tol,
        )

        if self.num_pieces > 1:
            self.sk_x_plus = torch.cat(
                [self.sk_x[..., 1:], self.qk_x_plus.unsqueeze(dim=-1)], dim=-1
            )
        else:
            self.sk_x_plus = self.qk_x_plus.unsqueeze(dim=-1)

        # Get tails parameters
        self.tail_al, self.tail_bl = ISQF.parameterize_tail(
            self.beta_l, self.qk_x_l, self.qk_y_l
        )
        self.tail_ar, self.tail_br = ISQF.parameterize_tail(
            -self.beta_r, 1 - self.qk_x_r, self.qk_y_r
        )

    @staticmethod
    def parameterize_qk(
        quantile_knots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Function to parameterize the x or y positions
        of the num_qk quantile knots
        Parameters
        ----------
        quantile_knots
            x or y positions of the quantile knots
            shape: (*batch_shape, num_qk)
        Returns
        -------
        qk
            x or y positions of the quantile knots (qk),
            with index=1, ..., num_qk-1,
            shape: (*batch_shape, num_qk-1)
        qk_plus
            x or y positions of the quantile knots (qk),
            with index=2, ..., num_qk,
            shape: (*batch_shape, num_qk-1)
        qk_l
            x or y positions of the left-most quantile knot (qk),
            shape: (*batch_shape)
        qk_r
            x or y positions of the right-most quantile knot (qk),
            shape: (*batch_shape)
        """

        qk, qk_plus = quantile_knots[..., :-1], quantile_knots[..., 1:]
        qk_l, qk_r = quantile_knots[..., 0], quantile_knots[..., -1]

        return qk, qk_plus, qk_l, qk_r

    @staticmethod
    def parameterize_spline(
        spline_knots: torch.Tensor,
        qk: torch.Tensor,
        qk_plus: torch.Tensor,
        tol: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Function to parameterize the x or y positions of the spline knots
        Parameters
        ----------
        spline_knots
            variable that parameterizes the spline knot positions
        qk
            x or y positions of the quantile knots (qk),
            with index=1, ..., num_qk-1,
            shape: (*batch_shape, num_qk-1)
        qk_plus
            x or y positions of the quantile knots (qk),
            with index=2, ..., num_qk,
            shape: (*batch_shape, num_qk-1)
        num_pieces
            number of spline knot pieces
        tol
            tolerance hyperparameter for numerical stability
        Returns
        -------
        sk
            x or y positions of the spline knots (sk),
            shape: (*batch_shape, num_qk-1, num_pieces)
        delta_sk
            difference of x or y positions of the spline knots (sk),
            shape: (*batch_shape, num_qk-1, num_pieces)
        """

        # The spacing between spline knots is parameterized
        # by softmax function (in [0,1] and sum to 1)
        # We add tol to prevent overflow in computing 1/spacing in spline CRPS
        # After adding tol, it is normalized by
        # (1 + num_pieces * tol) to keep the sum-to-1 property

        num_pieces = spline_knots.shape[-1]

        delta_x = (F.softmax(spline_knots, dim=-1) + tol) / (
            1 + num_pieces * tol
        )

        zero_tensor = torch.zeros_like(
            delta_x[..., 0:1]
        )  # 0:1 for keeping dimension
        x = torch.cat(
            [zero_tensor, torch.cumsum(delta_x, dim=-1)[..., :-1]], dim=-1
        )

        qk, qk_plus = qk.unsqueeze(dim=-1), qk_plus.unsqueeze(dim=-1)
        sk = x * (qk_plus - qk) + qk
        delta_sk = delta_x * (qk_plus - qk)

        return sk, delta_sk

    @staticmethod
    def parameterize_tail(
        beta: torch.Tensor, qk_x: torch.Tensor, qk_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Function to parameterize the tail parameters
        Note that the exponential tails are given by
        q(alpha)
        = a_l log(alpha) + b_l if left tail
        = a_r log(1-alpha) + b_r if right tail
        where
        a_l=1/beta_l, b_l=-a_l*log(qk_x_l)+q(qk_x_l)
        a_r=1/beta_r, b_r=a_r*log(1-qk_x_r)+q(qk_x_r)
        Parameters
        ----------
        beta
            parameterizes the left or right tail, shape: (*batch_shape,)
        qk_x
            left- or right-most x-positions of the quantile knots,
            shape: (*batch_shape,)
        qk_y
            left- or right-most y-positions of the quantile knots,
            shape: (*batch_shape,)
        Returns
        -------
        tail_a
            a_l or a_r as described above
        tail_b
            b_l or b_r as described above
        """

        tail_a = 1 / beta
        tail_b = -tail_a * torch.log(qk_x) + qk_y

        return tail_a, tail_b

    def quantile(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.quantile_internal(alpha, dim=0)

    def quantile_internal(
        self, alpha: torch.Tensor, dim: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Evaluates the quantile function at the quantile levels input_alpha
        Parameters
        ----------
        alpha
            Tensor of shape = (*batch_shape,) if axis=None, or containing an
            additional axis on the specified position, otherwise
        dim
            Index of the axis containing the different quantile levels which
            are to be computed.
            Read the description below for detailed information
        Returns
        -------
        Tensor
            Quantiles tensor, of the same shape as alpha
        """

        qk_x, qk_x_l, qk_x_plus = self.qk_x, self.qk_x_l, self.qk_x_plus

        # The following describes the parameters reshaping in
        # quantile_internal, quantile_spline and quantile_tail

        # tail parameters: tail_al, tail_ar, tail_bl, tail_br,
        # shape = (*batch_shape,)
        # spline parameters: sk_x, sk_x_plus, sk_y, sk_y_plus,
        # shape = (*batch_shape, num_qk-1, num_pieces)
        # quantile knots parameters: qk_x, qk_x_plus, qk_y, qk_y_plus,
        # shape = (*batch_shape, num_qk-1)

        # dim=None - passed at inference when num_samples is None
        # shape of input_alpha = (*batch_shape,), will be expanded to
        # (*batch_shape, 1, 1) to perform operation
        # The shapes of parameters are as described above,
        # no reshaping is needed

        # dim=0 - passed at inference when num_samples is not None
        # shape of input_alpha = (num_samples, *batch_shape)
        # it will be expanded to
        # (num_samples, *batch_shape, 1, 1) to perform operation
        #
        # The shapes of tail parameters
        # should be (num_samples, *batch_shape)
        #
        # The shapes of spline parameters
        # should be (num_samples, *batch_shape, num_qk-1, num_pieces)
        #
        # The shapes of quantile knots parameters
        # should be (num_samples, *batch_shape, num_qk-1)
        #
        # We expand at dim=0 for all of them

        # dim=-2 - passed at training when we evaluate quantiles at
        # spline knots in order to compute alpha_tilde
        #
        # This is only for the quantile_spline function
        # shape of input_alpha = (*batch_shape, num_qk-1, num_pieces)
        # it will be expanded to
        # (*batch_shape, num_qk-1, num_pieces, 1) to perform operation
        #
        # The shapes of spline and quantile knots parameters should be
        # (*batch_shape, num_qk-1, 1, num_pieces)
        # and (*batch_shape, num_qk-1, 1), respectively
        #
        # We expand at dim=-2 and dim=-1 for
        # spline and quantile knots parameters, respectively

        if dim is not None:
            qk_x_l = qk_x_l.unsqueeze(dim=dim)
            qk_x = qk_x.unsqueeze(dim=dim)
            qk_x_plus = qk_x_plus.unsqueeze(dim=dim)

        quantile = torch.where(
            alpha < qk_x_l,
            self.quantile_tail(alpha, dim=dim, left_tail=True),
            self.quantile_tail(alpha, dim=dim, left_tail=False),
        )

        spline_val = self.quantile_spline(alpha, dim=dim)

        for spline_idx in range(self.num_qk - 1):
            is_in_between = torch.logical_and(
                qk_x[..., spline_idx] <= alpha,
                alpha < qk_x_plus[..., spline_idx],
            )

            quantile = torch.where(
                is_in_between,
                spline_val[..., spline_idx],
                quantile,
            )

        return quantile

    def quantile_spline(
        self,
        alpha: torch.Tensor,
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        # Refer to the description in quantile_internal

        qk_y = self.qk_y
        sk_x, delta_sk_x, delta_sk_y = (
            self.sk_x,
            self.delta_sk_x,
            self.delta_sk_y,
        )

        if dim is not None:
            qk_y = qk_y.unsqueeze(dim=0 if dim == 0 else -1)
            sk_x = sk_x.unsqueeze(dim=dim)
            delta_sk_x = delta_sk_x.unsqueeze(dim=dim)
            delta_sk_y = delta_sk_y.unsqueeze(dim=dim)

        if dim is None or dim == 0:
            alpha = alpha.unsqueeze(dim=-1)

        alpha = alpha.unsqueeze(dim=-1)

        spline_val = (alpha - sk_x) / delta_sk_x
        spline_val = torch.maximum(
            torch.minimum(spline_val, torch.ones_like(spline_val)),
            torch.zeros_like(spline_val),
        )

        return qk_y + torch.sum(spline_val * delta_sk_y, dim=-1)

    def quantile_tail(
        self,
        alpha: torch.Tensor,
        dim: Optional[int] = None,
        left_tail: bool = True,
    ) -> torch.Tensor:
        # Refer to the description in quantile_internal

        if left_tail:
            tail_a, tail_b = self.tail_al, self.tail_bl
        else:
            tail_a, tail_b = self.tail_ar, self.tail_br
            alpha = 1 - alpha

        if dim is not None:
            tail_a, tail_b = tail_a.unsqueeze(dim=dim), tail_b.unsqueeze(
                dim=dim
            )

        return tail_a * torch.log(alpha) + tail_b

    def cdf_spline(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        For observations z and splines defined in [qk_x[k], qk_x[k+1]]
        Computes the quantile level alpha_tilde such that
        alpha_tilde
        = q^{-1}(z) if z is in-between qk_x[k] and qk_x[k+1]
        = qk_x[k] if z<qk_x[k]
        = qk_x[k+1] if z>qk_x[k+1]
        Parameters
        ----------
        z
            Observation, shape = (*batch_shape,)
        Returns
        -------
        alpha_tilde
            Corresponding quantile level, shape = (*batch_shape, num_qk-1)
        """

        qk_y, qk_y_plus = self.qk_y, self.qk_y_plus
        qk_x, qk_x_plus = self.qk_x, self.qk_x_plus
        sk_x, delta_sk_x, delta_sk_y = (
            self.sk_x,
            self.delta_sk_x,
            self.delta_sk_y,
        )

        z_expand = z.unsqueeze(dim=-1)

        if self.num_pieces > 1:
            qk_y_expand = qk_y.unsqueeze(dim=-1)
            z_expand_twice = z_expand.unsqueeze(dim=-1)

            knots_eval = self.quantile_spline(sk_x, dim=-2)

            # Compute \sum_{s=0}^{s_0-1} \Delta sk_y[s],
            # where \Delta sk_y[s] = (sk_y[s+1]-sk_y[s])
            mask_sum_s0 = torch.lt(knots_eval, z_expand_twice)
            mask_sum_s0_minus = torch.cat(
                [
                    mask_sum_s0[..., 1:],
                    torch.zeros_like(qk_y_expand, dtype=bool),
                ],
                dim=-1,
            )
            sum_delta_sk_y = torch.sum(mask_sum_s0_minus * delta_sk_y, dim=-1)

            mask_s0_only = torch.logical_and(
                mask_sum_s0, torch.logical_not(mask_sum_s0_minus)
            )
            # Compute (sk_x[s_0+1]-sk_x[s_0])/(sk_y[s_0+1]-sk_y[s_0])
            frac_s0 = torch.sum(
                (mask_s0_only * delta_sk_x) / delta_sk_y, dim=-1
            )

            # Compute sk_x_{s_0}
            sk_x_s0 = torch.sum(mask_s0_only * sk_x, dim=-1)

            # Compute alpha_tilde
            alpha_tilde = (
                sk_x_s0 + (z_expand - qk_y - sum_delta_sk_y) * frac_s0
            )

        else:
            # num_pieces=1, ISQF reduces to IQF
            alpha_tilde = qk_x + (z_expand - qk_y) / (qk_y_plus - qk_y) * (
                qk_x_plus - qk_x
            )

        alpha_tilde = torch.minimum(
            torch.maximum(alpha_tilde, qk_x), qk_x_plus
        )

        return alpha_tilde

    def cdf_tail(
        self, z: torch.Tensor, left_tail: bool = True
    ) -> torch.Tensor:
        r"""
        Computes the quantile level alpha_tilde such that
        alpha_tilde
        = q^{-1}(z) if z is in the tail region
        = qk_x_l or qk_x_r if z is in the non-tail region
        Parameters
        ----------
        z
            Observation, shape = (*batch_shape,)
        left_tail
            If True, compute alpha_tilde for the left tail
            Otherwise, compute alpha_tilde for the right tail
        Returns
        -------
        alpha_tilde
            Corresponding quantile level, shape = (*batch_shape,)
        """

        if left_tail:
            tail_a, tail_b, qk_x = self.tail_al, self.tail_bl, self.qk_x_l
        else:
            tail_a, tail_b, qk_x = self.tail_ar, self.tail_br, 1 - self.qk_x_r

        log_alpha_tilde = torch.minimum((z - tail_b) / tail_a, torch.log(qk_x))
        alpha_tilde = torch.exp(log_alpha_tilde)
        return alpha_tilde if left_tail else 1 - alpha_tilde

    def crps_tail(
        self, z: torch.Tensor, left_tail: bool = True
    ) -> torch.Tensor:
        r"""
        Compute CRPS in analytical form for left/right tails
        Parameters
        ----------
        z
            Observation to evaluate. shape = (*batch_shape,)
        left_tail
            If True, compute CRPS for the left tail
            Otherwise, compute CRPS for the right tail
        Returns
        -------
        Tensor
            Tensor containing the CRPS, of the same shape as z
        """

        alpha_tilde = self.cdf_tail(z, left_tail=left_tail)

        if left_tail:
            tail_a, tail_b, qk_x, qk_y = (
                self.tail_al,
                self.tail_bl,
                self.qk_x_l,
                self.qk_y_l,
            )
            term1 = (z - tail_b) * (qk_x**2 - 2 * qk_x + 2 * alpha_tilde)
            term2 = qk_x**2 * tail_a * (-torch.log(qk_x) + 0.5)
            term2 = term2 + 2 * torch.where(
                z < qk_y,
                qk_x * tail_a * (torch.log(qk_x) - 1)
                + alpha_tilde * (-z + tail_b + tail_a),
                torch.zeros_like(qk_x),
            )
        else:
            tail_a, tail_b, qk_x, qk_y = (
                self.tail_ar,
                self.tail_br,
                self.qk_x_r,
                self.qk_y_r,
            )
            term1 = (z - tail_b) * (-1 - qk_x**2 + 2 * alpha_tilde)
            term2 = tail_a * (
                -0.5 * (qk_x + 1) ** 2
                + (qk_x**2 - 1) * torch.log(1 - qk_x)
                + 2 * alpha_tilde
            )
            term2 = term2 + 2 * torch.where(
                z > qk_y,
                (1 - alpha_tilde) * (z - tail_b),
                tail_a * (1 - qk_x) * torch.log(1 - qk_x),
            )

        return term1 + term2

    def crps_spline(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Compute CRPS in analytical form for the spline
        Parameters
        ----------
        z
            Observation to evaluate. shape = (*batch_shape,)
        Returns
        -------
        Tensor
            Tensor containing the CRPS, of the same shape as z
        """

        qk_x, qk_x_plus, qk_y = self.qk_x, self.qk_x_plus, self.qk_y
        sk_x, sk_x_plus = self.sk_x, self.sk_x_plus
        delta_sk_x, delta_sk_y = self.delta_sk_x, self.delta_sk_y

        z_expand = z.unsqueeze(dim=-1)
        qk_x_plus_expand = qk_x_plus.unsqueeze(dim=-1)

        alpha_tilde = self.cdf_spline(z)
        alpha_tilde_expand = alpha_tilde.unsqueeze(dim=-1)

        r = torch.minimum(torch.maximum(alpha_tilde_expand, sk_x), sk_x_plus)

        coeff1 = (
            -2 / 3 * sk_x_plus**3
            + sk_x * sk_x_plus**2
            + sk_x_plus**2
            - (1 / 3) * sk_x**3
            - 2 * sk_x * sk_x_plus
            - r**2
            + 2 * sk_x * r
        )

        coeff2 = (
            -2 * torch.maximum(alpha_tilde_expand, sk_x_plus)
            + sk_x_plus**2
            + 2 * qk_x_plus_expand
            - qk_x_plus_expand**2
        )

        result = (
            (qk_x_plus**2 - qk_x**2) * (z_expand - qk_y)
            + 2 * (qk_x_plus - alpha_tilde) * (qk_y - z_expand)
            + torch.sum((delta_sk_y / delta_sk_x) * coeff1, dim=-1)
            + torch.sum(delta_sk_y * coeff2, dim=-1)
        )

        return torch.sum(result, dim=-1)

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.crps(z)

    def crps(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Compute CRPS in analytical form
        Parameters
        ----------
        z
            Observation to evaluate. Shape = (*batch_shape,)
        Returns
        -------
        Tensor
            Tensor containing the CRPS, of the same shape as z
        """

        crps_lt = self.crps_tail(z, left_tail=True)
        crps_rt = self.crps_tail(z, left_tail=False)

        return crps_lt + crps_rt + self.crps_spline(z)

    def cdf(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the quantile level alpha_tilde such that
        q(alpha_tilde) = z
        Parameters
        ----------
        z
            Tensor of shape = (*batch_shape,)
        Returns
        -------
        alpha_tilde
            Tensor of shape = (*batch_shape,)
        """

        qk_y, qk_y_l, qk_y_plus = self.qk_y, self.qk_y_l, self.qk_y_plus

        alpha_tilde = torch.where(
            z < qk_y_l,
            self.cdf_tail(z, left_tail=True),
            self.cdf_tail(z, left_tail=False),
        )

        spline_alpha_tilde = self.cdf_spline(z)

        for spline_idx in range(self.num_qk - 1):
            is_in_between = torch.logical_and(
                qk_y[..., spline_idx] <= z, z < qk_y_plus[..., spline_idx]
            )

            alpha_tilde = torch.where(
                is_in_between, spline_alpha_tilde[..., spline_idx], alpha_tilde
            )

        return alpha_tilde

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Function used to draw random samples
        Parameters
        ----------
        num_samples
            number of samples
        dtype
            data type
        Returns
        -------
        Tensor
            Tensor of shape (*batch_shape,) if num_samples = None
            else (num_samples, *batch_shape)
        """

        # if sample_shape=()) then input_alpha should have the same shape
        # as beta_l, i.e., (*batch_shape,)
        # else u should be (*sample_shape, *batch_shape)
        target_shape = (
            self.beta_l.shape
            if sample_shape == torch.Size()
            else torch.Size(sample_shape) + self.beta_l.shape
        )

        alpha = torch.rand(
            target_shape,
            dtype=self.beta_l.dtype,
            device=self.beta_l.device,
            layout=self.beta_l.layout,
        )

        sample = self.quantile(alpha)

        if sample_shape == torch.Size():
            sample = sample.squeeze(dim=0)

        return sample

    @property
    def batch_shape(self) -> torch.Size():
        return self.beta_l.shape


class ISQFOutput(DistributionOutput):
    r"""
    DistributionOutput class for the Incremental (Spline) Quantile Function
    Parameters
    ----------
    num_pieces
        number of spline pieces for each spline
        ISQF reduces to IQF when num_pieces = 1
    qk_x
        List containing the x-positions of quantile knots
    tol
        tolerance for numerical safeguarding
    """

    distr_cls: type = ISQF

    @validated()
    def __init__(
        self, num_pieces: int, qk_x: List[float], tol: float = 1e-4
    ) -> None:
        # ISQF reduces to IQF when num_pieces = 1

        super().__init__(self)

        assert (
            isinstance(num_pieces, int) and num_pieces > 0
        ), "num_pieces should be an integer and greater than 0"

        self.num_pieces = num_pieces
        self.qk_x = sorted(qk_x)
        self.num_qk = len(qk_x)
        self.tol = tol
        self.args_dim: Dict[str, int] = {
            "spline_knots": (self.num_qk - 1) * num_pieces,
            "spline_heights": (self.num_qk - 1) * num_pieces,
            "beta_l": 1,
            "beta_r": 1,
            "quantile_knots": self.num_qk,
        }

    @classmethod
    def domain_map(
        cls,
        spline_knots: torch.Tensor,
        spline_heights: torch.Tensor,
        beta_l: torch.Tensor,
        beta_r: torch.Tensor,
        quantile_knots: torch.Tensor,
        tol: float = 1e-4,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Domain map function The inputs of this function are specified by
        self.args_dim.

        spline_knots, spline_heights:
        parameterizing the x-/ y-positions of the spline knots,
        shape = (*batch_shape, (num_qk-1)*num_pieces)

        beta_l, beta_r:
        parameterizing the left/right tail, shape = (*batch_shape, 1)

        quantile_knots:
        parameterizing the y-positions of the quantile knots,
        shape = (*batch_shape, num_qk)
        """

        # Add tol to prevent the y-distance of
        # two quantile knots from being too small
        #
        # Because in this case the spline knots could be squeezed together
        # and cause overflow in spline CRPS computation
        qk_y = torch.cat(
            [
                quantile_knots[..., 0:1],
                torch.abs(quantile_knots[..., 1:]) + tol,
            ],
            dim=-1,
        )
        qk_y = torch.cumsum(qk_y, dim=-1)

        # Prevent overflow when we compute 1/beta
        beta_l = torch.abs(beta_l.squeeze(-1)) + tol
        beta_r = torch.abs(beta_r.squeeze(-1)) + tol

        return spline_knots, spline_heights, beta_l, beta_r, qk_y

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> ISQF:
        """
        function outputing the distribution class
        distr_args: distribution arguments
        loc: shift to the data mean
        scale: scale to the data
        """

        distr_args, qk_x = self.reshape_spline_args(distr_args, self.qk_x)

        distr = self.distr_cls(*distr_args, qk_x, self.tol)

        if scale is None:
            return distr
        else:
            return TransformedISQF(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    def reshape_spline_args(self, distr_args, qk_x: List[float]):
        """
        auxiliary function reshaping knots and heights to (*batch_shape,
        num_qk-1, num_pieces) qk_x to (*batch_shape, num_qk)
        """

        spline_knots, spline_heights = distr_args[0], distr_args[1]
        batch_shape = spline_knots.shape[:-1]
        num_qk, num_pieces = self.num_qk, self.num_pieces

        # repeat qk_x from (num_qk,) to (*batch_shape, num_qk)
        qk_x_repeat = torch.tensor(
            qk_x, dtype=spline_knots.dtype, device=spline_knots.device
        ).repeat(*batch_shape, 1)

        # knots and heights have shape (*batch_shape, (num_qk-1)*num_pieces)
        # reshape them to (*batch_shape, (num_qk-1), num_pieces)
        spline_knots_reshape = spline_knots.reshape(
            *batch_shape, (num_qk - 1), num_pieces
        )
        spline_heights_reshape = spline_heights.reshape(
            *batch_shape, (num_qk - 1), num_pieces
        )

        distr_args_reshape = (
            spline_knots_reshape,
            spline_heights_reshape,
            *distr_args[2:],
        )

        return distr_args_reshape, qk_x_repeat

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedISQF(TransformedDistribution):
    # Distribution class for the case when loc/scale is not None
    @validated()
    def __init__(
        self,
        base_distribution: ISQF,
        transforms: List[AffineTransform],
        validate_args=None,
    ) -> None:
        super().__init__(
            base_distribution, transforms, validate_args=validate_args
        )

    def crps(self, y: torch.Tensor) -> torch.Tensor:
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(t, AffineTransform), "Not an AffineTransform"
            z = t._inverse(z)
            scale *= t.scale
        p = self.base_dist.crps(z)
        return p * scale

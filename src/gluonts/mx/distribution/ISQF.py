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

from typing import Dict, List, Optional, Tuple, cast

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.util import cumsum

from .bijection import AffineTransformation, Bijection
from .distribution import Distribution, getF
from .distribution_output import DistributionOutput
from .transformed_distribution import TransformedDistribution


class ISQF(Distribution):
    r"""
    Distribution class for the Incremental (Spline) Quantile Function in the paper
    ``Learning Quantile Functions without Quantile Crossing for Distribution-free Time Series Forecasting``
    by Park, Robinson, Aubet, Kan, Gasthaus, Wang

    Parameters
    ----------
    knots, heights
        Tensor parametrizing the x-positions (y-positions) of the spline knots
        Shape: ``(*batch_shape, (num_qk-1), num_pieces)``
    alpha, q_proj
        Tensor containing the increasing x-positions (y-positions) of the quantile knots
        Shape: ``(*batch_shape, num_qk)``
    beta_l, beta_r
        Tensor containing the non-negative learnable parameter of the left (right) tail
        Shape: ``(*batch_shape)``
    """
    is_reparameterizable = False

    @validated()
    def __init__(
        self,
        knots: Tensor,
        heights: Tensor,
        q_proj: Tensor,
        beta_l: Tensor,
        beta_r: Tensor,
        alpha: Tensor,
        num_qk: int,
        num_pieces: int,
    ) -> None:
        self.num_qk, self.num_pieces = num_qk, num_pieces
        self.knots, self.heights = knots, heights
        self.beta_l, self.beta_r = beta_l, beta_r
        self.q_proj = q_proj

        F = self.F

        # Get quantile knots parameters

        # alpha_k: alpha_k, q_k: q(alpha_k) (left quantile knot position for each spline)
        # alpha_kplus: alpha_{k+1}, q_kplus: q(alpha_{k+1}) (right quantile knot position for each spline)
        # for k=1,...,num_qk-1
        # shape=(*batch_size, num_qk-1)

        # alpha_l: alpha_0, q_l: q(alpha_0) (leftmost quantile knot position)
        # alpha_r: alpha_{num_qk}, q_r: q(alpha_{num_qk}) (rightmost quantile knot position)
        # shape=(*batch_size)
        (
            self.alpha,
            self.alpha_plus,
            self.alpha_l,
            self.alpha_r,
        ) = ISQF.parametrize_qk(F, alpha)
        (
            self.q,
            self.q_plus,
            self.q_l,
            self.q_r,
        ) = ISQF.parametrize_qk(F, q_proj)

        # Get spline knots parameters

        # p: p_s, delta_p: p_{s+1}-p_s
        # d: d_s, d_plus: d_{s+1}, delta_d: d_{s+1}-d_s
        # for s=0,...,num_pieces-1
        # shape=(*batch_size, num_qk-1, num_pieces)
        self.p, self.delta_p = ISQF.parametrize_spline(
            F, self.heights, self.q, self.q_plus, self.num_pieces
        )
        self.d, self.delta_d = ISQF.parametrize_spline(
            F, self.knots, self.alpha, self.alpha_plus, self.num_pieces
        )

        if self.num_pieces > 1:
            self.d_plus = F.concat(
                F.slice_axis(self.d, axis=-1, begin=1, end=None),
                F.expand_dims(self.alpha_plus, axis=-1),
                dim=-1,
            )
        else:
            self.d_plus = F.expand_dims(self.alpha_plus, axis=-1)

        # Get tails parameters
        self.a_l, self.b_l = ISQF.parametrize_tail(
            F, self.beta_l, self.alpha_l, self.q_l
        )
        self.a_r, self.b_r = ISQF.parametrize_tail(
            F, -self.beta_r, 1 - self.alpha_r, self.q_r
        )

    @property
    def F(self):
        return getF(self.beta_l)

    @property
    def args(self) -> List:
        return [
            self.knots,
            self.heights,
            self.q_proj,
            self.beta_l,
            self.beta_r,
        ]

    @staticmethod
    def parametrize_qk(
        F, q_proj: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        q = F.slice_axis(q_proj, axis=-1, begin=0, end=-1)
        q_plus = F.slice_axis(q_proj, axis=-1, begin=1, end=None)
        q_l = F.slice_axis(q_proj, axis=-1, begin=0, end=1).squeeze(axis=-1)
        q_r = F.slice_axis(q_proj, axis=-1, begin=-1, end=None).squeeze(
            axis=-1
        )

        return q, q_plus, q_l, q_r

    @staticmethod
    def parametrize_spline(
        F, knots: Tensor, alpha: Tensor, alpha_plus: Tensor, num_pieces: int
    ) -> Tuple[Tensor, Tensor]:
        # For numerical stability in CRPS computation
        delta_x = (F.softmax(knots) + 1e-4) / (1 + num_pieces * 1e-4)
        x = cumsum(F, delta_x, exclusive=True)

        alpha = F.expand_dims(alpha, axis=-1)
        alpha_plus = F.expand_dims(alpha_plus, axis=-1)

        d = F.broadcast_add(F.broadcast_mul(x, (alpha_plus - alpha)), alpha)
        delta_d = F.broadcast_mul(delta_x, (alpha_plus - alpha))

        return d, delta_d

    @staticmethod
    def parametrize_tail(
        F, beta: Tensor, alpha: Tensor, q: Tensor
    ) -> Tuple[Tensor, Tensor]:
        a = 1 / beta
        b = -a * F.log(alpha) + q

        return a, b

    def quantile(self, input_alpha: Tensor) -> Tensor:
        return self.quantile_internal(input_alpha, axis=0)

    def quantile_internal(
        self, input_alpha: Tensor, axis: Optional[int] = None
    ) -> Tensor:
        r"""
        Evaluates the quantile function at the quantile levels input_alpha.
        Parameters
        ----------
        input_alpha
            Tensor of shape ``(*beta_l.shape,)`` if axis=None, or containing an
            additional axis on the specified position, otherwise.
        axis
            Index of the axis containing the different quantile levels which
            are to be computed. Read the description below for detailed information
        Returns
        -------
        Tensor
            Quantiles tensor, of the same shape as input_alpha.
        """

        # The following describes the parameters reshaping in quantile_internal, quantile_spline and quantile_tail

        # tail parameters: a_l, a_r, b_l, b_r
        # shape = (*batch_shape)

        # spline parameters: d, d_plus, p, p_plus
        # shape = (*batch_shape, num_qk-1, num_pieces)

        # quantile knots parameters: alpha, alpha_plus, q, q_plus
        # shape = (*batch_shape, num_qk-1)

        # axis=None - passed at inference when num_samples is None
        # shape of input_alpha = (*batch_shape)
        # it will be expanded to (*batch_shape, 1, 1) to perform operation
        # The shapes of parameters are as described above, no reshaping is needed

        # axis=0 - passed at inference when num_samples is not None
        # shape of input_alpha = (num_samples, *batch_shape)
        # it will be expanded to (num_samples, *batch_shape, 1, 1) to perform operation
        # The shapes of tail parameters should be (num_samples, *batch_shape)
        # The shapes of spline parameters should be (num_samples, *batch_shape, num_qk-1, num_pieces)
        # The shapes of quantile knots parameters should be (num_samples, *batch_shape, num_qk-1)
        # We expand axis=0 for all of them

        # axis=-2 - passed at training when we evaluate quantiles at spline knots in order to compute alpha_tilde
        # This is only for the quantile_spline function
        # shape of input_alpha = (*batch_shape, num_qk-1, num_pieces)
        # it will be expanded to (*batch_shape, num_qk-1, num_pieces, 1) to perform operation
        # The shapes of the spline parameters should be (*batch_shape, num_qk-1, 1, num_pieces)
        # The shapes of quantile knots parameters should be (*batch_shape, num_qk-1, 1)
        # We expand axis=-1 and axis=-2 for quantile knots and spline parameters, respectively

        F = self.F
        alpha, alpha_l, alpha_plus = self.alpha, self.alpha_l, self.alpha_plus

        if axis is not None:
            alpha_l = F.expand_dims(alpha_l, axis=axis)
            alpha = F.expand_dims(alpha, axis=axis)
            alpha_plus = F.expand_dims(alpha_plus, axis=axis)

        quantile = F.where(
            input_alpha < alpha_l,
            self.quantile_tail(input_alpha, axis=axis, left_tail=True),
            self.quantile_tail(input_alpha, axis=axis, left_tail=False),
        )

        spline_val = self.quantile_spline(input_alpha, axis=axis)

        for i in range(self.num_qk - 1):

            is_in_between = F.broadcast_logical_and(
                F.slice_axis(alpha, axis=-1, begin=i, end=i + 1).squeeze(-1)
                <= input_alpha,
                input_alpha
                < F.slice_axis(
                    alpha_plus, axis=-1, begin=i, end=i + 1
                ).squeeze(-1),
            )

            quantile = F.where(
                is_in_between,
                F.slice_axis(spline_val, axis=-1, begin=i, end=i + 1).squeeze(
                    -1
                ),
                quantile,
            )

        return quantile

    def quantile_spline(
        self,
        input_alpha: Tensor,
        axis: Optional[int] = None,
    ) -> Tensor:
        # Refer to the description in quantile_internal

        F = self.F
        q = self.q
        d, delta_d, delta_p = self.d, self.delta_d, self.delta_p

        if axis is not None:
            q = F.expand_dims(q, axis=0 if axis == 0 else -1)
            d = F.expand_dims(d, axis=axis)
            delta_d = F.expand_dims(delta_d, axis=axis)
            delta_p = F.expand_dims(delta_p, axis=axis)

        if axis is None or axis == 0:
            input_alpha = F.expand_dims(input_alpha, axis=-1)

        input_alpha = F.expand_dims(input_alpha, axis=-1)

        spline_val = F.broadcast_div(F.broadcast_sub(input_alpha, d), delta_d)
        spline_val = F.maximum(
            F.minimum(spline_val, F.ones_like(spline_val)),
            F.zeros_like(spline_val),
        )

        return F.broadcast_add(
            q,
            F.sum(
                F.broadcast_mul(spline_val, delta_p), axis=-1, keepdims=False
            ),
        )

    def quantile_tail(
        self,
        input_alpha: Tensor,
        axis: Optional[int] = None,
        left_tail: bool = True,
    ) -> Tensor:
        # Refer to the description in quantile_internal

        F = self.F

        if left_tail:
            a, b = self.a_l, self.b_l
        else:
            a, b = self.a_r, self.b_r
            input_alpha = 1 - input_alpha

        if axis is not None:
            a, b = F.expand_dims(a, axis=axis), F.expand_dims(b, axis=axis)

        return F.broadcast_add(F.broadcast_mul(a, F.log(input_alpha)), b)

    def get_alpha_tilde_spline(self, z: Tensor) -> Tensor:
        # For a spline defined in [alpha_k, alpha_{k+1}]
        # Computes the quantile level alpha_tilde such that
        # alpha_tilde
        # = q^{-1}(z) if z is in-between q_k and q_{k+1}
        # = alpha_k if z<q(alpha_k)
        # = alpha_{k+1} if z>q(alpha_{k+1})

        F = self.F

        q, q_plus = self.q, self.q_plus
        alpha, alpha_plus = self.alpha, self.alpha_plus
        d, delta_d, delta_p = self.d, self.delta_d, self.delta_p

        z_expand = F.expand_dims(z, axis=-1)

        if self.num_pieces > 1:
            q_expand = F.expand_dims(q, axis=-1)
            z_expand_twice = F.expand_dims(z_expand, axis=-1)

            knots_eval = self.quantile_spline(d, axis=-2)

            # Compute \sum_{s=0}^{s_0-1} \Delta p_s, where \Delta p_s = (p_{s+1}-p_s)
            mask_sum_s0 = F.broadcast_lesser(knots_eval, z_expand_twice)
            mask_sum_s0_minus = F.concat(
                F.slice_axis(mask_sum_s0, axis=-1, begin=1, end=None),
                F.zeros_like(q_expand),
                dim=-1,
            )
            sum_delta_p = F.sum(
                F.broadcast_mul(mask_sum_s0_minus, delta_p),
                axis=-1,
                keepdims=False,
            )

            # Compute (d_{s_0+1}-d_{s_0})/(p_{s_0+1}-p_{s_0})
            mask_s0_only = mask_sum_s0 - mask_sum_s0_minus
            # (\Delta d_{s_0} / \Delta h_{s_0})
            delta_d_div_p_s0 = F.sum(
                (mask_s0_only * delta_d) / delta_p, axis=-1, keepdims=False
            )

            # Compute d_{s_0}
            d_s0 = F.sum(mask_s0_only * d, axis=-1, keepdims=False)

            # Compute alpha_tilde
            alpha_tilde = (
                d_s0
                + (F.broadcast_sub(z_expand, q) - sum_delta_p)
                * delta_d_div_p_s0
            )

        else:
            # num_pieces=1, ISQF reduces to IQF
            alpha_tilde = alpha + F.broadcast_sub(z_expand, q) / (
                q_plus - q
            ) * (alpha_plus - alpha)

        alpha_tilde = F.broadcast_minimum(
            F.broadcast_maximum(alpha_tilde, alpha), alpha_plus
        )

        return alpha_tilde

    def get_alpha_tilde_tail(
        self, z: Tensor, left_tail: bool = True
    ) -> Tensor:
        # Computes the quantile level alpha_tilde such that
        # alpha_tilde
        # = q^{-1}(z) if z is in the tail region
        # = alpha_tail if z is in the non-tail region

        F = self.F

        if left_tail:
            a, b, alpha = self.a_l, self.b_l, self.alpha_l
        else:
            a, b, alpha = self.a_r, self.b_r, 1 - self.alpha_r

        log_alpha_tilde = F.minimum((z - b) / a, F.log(alpha))
        alpha_tilde = F.exp(log_alpha_tilde)
        return alpha_tilde if left_tail else 1 - alpha_tilde

    def crps_tail(self, z: Tensor, left_tail: bool = True) -> Tensor:
        F = self.F
        alpha_tilde = self.get_alpha_tilde_tail(z, left_tail=left_tail)

        if left_tail:
            a, b, alpha, q = self.a_l, self.b_l, self.alpha_l, self.q_l
            term1 = (z - b) * (alpha ** 2 - 2 * alpha + 2 * alpha_tilde)
            term2 = alpha ** 2 * a * (-F.log(alpha) + 0.5)
            term2 = term2 + 2 * F.where(
                z < q,
                alpha * a * (F.log(alpha) - 1) + alpha_tilde * (-z + b + a),
                F.zeros_like(alpha),
            )
        else:
            a, b, alpha, q = self.a_r, self.b_r, self.alpha_r, self.q_r
            term1 = (z - b) * (-1 - alpha ** 2 + 2 * alpha_tilde)
            term2 = a * (
                -0.5 * (alpha + 1) ** 2
                + (alpha ** 2 - 1) * F.log(1 - alpha)
                + 2 * alpha_tilde
            )
            term2 = term2 + 2 * F.where(
                z > q,
                (1 - alpha_tilde) * (z - b),
                a * (1 - alpha) * F.log(1 - alpha),
            )

        result = term1 + term2

        return result

    def crps_spline(self, z: Tensor) -> Tensor:

        F = self.F
        alpha, alpha_plus, q = self.alpha, self.alpha_plus, self.q
        d, d_plus = self.d, self.d_plus
        delta_d, delta_p = self.delta_d, self.delta_p

        z_expand, alpha_plus_expand = F.expand_dims(z, axis=-1), F.expand_dims(
            alpha_plus, axis=-1
        )

        alpha_tilde = self.get_alpha_tilde_spline(z)
        alpha_tilde_expand = F.expand_dims(alpha_tilde, axis=-1)

        r = F.broadcast_minimum(
            F.broadcast_maximum(alpha_tilde_expand, d), d_plus
        )

        coeff1 = (
            -2 / 3 * d_plus ** 3
            + d * d_plus ** 2
            + d_plus ** 2
            - (1 / 3) * d ** 3
            - 2 * d * d_plus
            - r ** 2
            + 2 * d * r
        )

        coeff2 = F.broadcast_add(
            -2 * F.broadcast_maximum(alpha_tilde_expand, d_plus) + d_plus ** 2,
            2 * alpha_plus_expand - alpha_plus_expand ** 2,
        )

        result = (
            (alpha_plus ** 2 - alpha ** 2) * F.broadcast_sub(z_expand, q)
            + 2
            * F.broadcast_sub(alpha_plus, alpha_tilde)
            * F.broadcast_sub(q, z_expand)
            + F.sum((delta_p / delta_d) * coeff1, axis=-1, keepdims=False)
            + F.sum(delta_p * coeff2, axis=-1, keepdims=False)
        )

        return F.sum(result, axis=-1, keepdims=False)

    def loss(self, z: Tensor) -> Tensor:
        return self.crps(z)

    def crps(self, z: Tensor) -> Tensor:
        r"""
        Compute CRPS in analytical form.
        Parameters
        ----------
        z
            Observation to evaluate. Shape equals to beta_l.shape.
        Returns
        -------
        Tensor
            Tensor containing the CRPS.
        """

        crps_lt = self.crps_tail(z, left_tail=True)
        crps_rt = self.crps_tail(z, left_tail=False)

        return crps_lt + crps_rt + self.crps_spline(z)

    def cdf(self, z: Tensor) -> Tensor:
        r"""
        Computes the quantile level alpha_tilde such that
        q(alpha_tilde) = z
        Parameters
        ----------
        z
            Tensor of shape beta_l.shape
        Returns
        -------
        Tensor
            Tensor of shape beta_l.shape
        """
        F = self.F
        q, q_l, q_plus = self.q, self.q_l, self.q_plus

        alpha_tilde = F.where(
            z < q_l,
            self.get_alpha_tilde_tail(z, left_tail=True),
            self.get_alpha_tilde_tail(z, left_tail=False),
        )

        spline_alpha_tilde = self.get_alpha_tilde_spline(z)

        for i in range(self.num_qk - 1):

            is_in_between = F.broadcast_logical_and(
                F.slice_axis(q, axis=-1, begin=i, end=i + 1).squeeze(-1) <= z,
                z
                < F.slice_axis(q_plus, axis=-1, begin=i, end=i + 1).squeeze(
                    -1
                ),
            )

            alpha_tilde = F.where(
                is_in_between,
                F.slice_axis(
                    spline_alpha_tilde, axis=-1, begin=i, end=i + 1
                ).squeeze(-1),
                alpha_tilde,
            )

        return alpha_tilde

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:

        F = self.F

        # if num_samples=None then input_alpha should have the same shape as beta_l, i.e., (*batch_shape,)
        # else u should be (num_samples, *batch_shape)
        input_alpha = F.random.uniform_like(
            data=(
                self.beta_l
                if num_samples is None
                else self.beta_l.expand_dims(axis=0).repeat(
                    axis=0, repeats=num_samples
                )
            )
        )

        sample = self.quantile(input_alpha)

        if num_samples is None:
            sample = F.squeeze(sample, axis=0)

        return sample

    @property
    def batch_shape(self) -> Tuple:
        return self.beta_l.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0


class ISQFOutput(DistributionOutput):
    distr_cls: type = ISQF

    @validated()
    def __init__(self, num_pieces: int, alpha: List[float]) -> None:
        # ISQF reduces to IQF when num_pieces = 1

        super().__init__(self)

        assert (
            isinstance(num_pieces, int) and num_pieces > 0
        ), "num_pieces should be an integer and greater than 0"

        self.num_pieces = num_pieces
        self.num_qk = len(alpha)
        self.alpha = alpha
        self.args_dim = cast(
            Dict[str, int],
            {
                "knots": (self.num_qk - 1) * num_pieces,
                "heights": (self.num_qk - 1) * num_pieces,
                "q": self.num_qk,
                "beta_l": 1,
                "beta_r": 1,
            },
        )

    @classmethod
    def domain_map(
        cls,
        F,
        knots: Tensor,
        heights: Tensor,
        q: Tensor,
        beta_l: Tensor,
        beta_r: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        q_proj = F.concat(
            F.slice_axis(q, axis=-1, begin=0, end=1),
            F.abs(F.slice_axis(q, axis=-1, begin=1, end=None)) + 1e-4,
            dim=-1,
        )
        q_proj = cumsum(F, q_proj)

        beta_l, beta_r = (
            F.abs(beta_l.squeeze(axis=-1)) + 1e-4,
            F.abs(beta_r.squeeze(axis=-1)) + 1e-4,
        )

        return knots, heights, q_proj, beta_l, beta_r

    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> ISQF:

        distr_args, alpha = self.reshape_spline_args(distr_args, self.alpha)

        if scale is None:
            return self.distr_cls(
                *distr_args, alpha, self.num_qk, self.num_pieces
            )
        else:
            distr = self.distr_cls(
                *distr_args, alpha, self.num_qk, self.num_pieces
            )
            return TransformedISQF(
                distr, [AffineTransformation(loc=loc, scale=scale)]
            )

    def reshape_spline_args(self, distr_args, alpha):
        knots, heights = distr_args[0], distr_args[1]
        beta_l = distr_args[3]
        # can be deleted after reshape issue has been resolved
        q = distr_args[2]
        F = getF(beta_l)

        # FIXME number 1
        # Convert alpha from list of len=num_qk to Tensor of shape (*batch_shape, num_qk)
        # For example, if alpha = [0.1, 0.5, 0.9],
        # then alpha_reshape will be a Tensor of shape (*batch_shape, 3) with the last dimension being [0.1, 0.5, 0.9]
        # In PyTorch, it would be torch.tensor(alpha).repeat(*batch_shape,1)
        alpha_reshape = F.concat(
            *[
                F.expand_dims(F.ones_like(beta_l), axis=-1) * alpha[i]
                for i in range(self.num_qk)
            ],
            dim=-1
        )

        # FIXME number 2
        # knots and heights have shape (*batch_shape, (num_qk-1)*num_pieces)
        # I want to convert the shape to (*batch_shape, (num_qk-1), num_pieces)
        # Here I make a shape_holder with the target shape, and use reshape_like

        # create a shape holder of shape (*batch_size, num_qk-1, num_pieces)
        shape_holder = F.slice_axis(q, axis=-1, begin=0, end=-1)
        shape_holder = F.expand_dims(shape_holder, axis=-1)
        shape_holder = F.repeat(shape_holder, repeats=self.num_pieces, axis=-1)

        knots_reshape = F.reshape_like(knots, shape_holder)
        heights_reshape = F.reshape_like(heights, shape_holder)
        distr_args_reshape = (knots_reshape, heights_reshape) + distr_args[2:]

        return distr_args_reshape, alpha_reshape

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedISQF(TransformedDistribution, ISQF):
    @validated()
    def __init__(
        self, base_distribution: ISQF, transforms: List[Bijection]
    ) -> None:

        super().__init__(base_distribution, transforms)

    def crps(self, y: Tensor) -> Tensor:
        F = getF(y)
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(
                t, AffineTransformation
            ), "Not an AffineTransformation"
            z = t.f_inv(z)
            scale *= t.scale
        p = self.base_distribution.crps(z)
        return F.broadcast_mul(p, scale)

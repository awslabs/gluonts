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

    is_reparameterizable = False

    @validated()
    def __init__(
        self,
        spline_knots: Tensor,
        spline_heights: Tensor,
        beta_l: Tensor,
        beta_r: Tensor,
        qk_y: Tensor,
        qk_x: Tensor,
        num_qk: int,
        num_pieces: int,
        tol: float = 1e-4,
    ) -> None:
        self.num_qk, self.num_pieces = num_qk, num_pieces
        self.spline_knots, self.spline_heights = spline_knots, spline_heights
        self.beta_l, self.beta_r = beta_l, beta_r
        self.qk_y_all = qk_y
        self.tol = tol

        F = self.F

        # Get quantile knots (qk) parameters
        (
            self.qk_x,
            self.qk_x_plus,
            self.qk_x_l,
            self.qk_x_r,
        ) = ISQF.parametrize_qk(F, qk_x)
        (
            self.qk_y,
            self.qk_y_plus,
            self.qk_y_l,
            self.qk_y_r,
        ) = ISQF.parametrize_qk(F, qk_y)

        # Get spline knots (sk) parameters
        self.sk_y, self.delta_sk_y = ISQF.parametrize_spline(
            F,
            self.spline_heights,
            self.qk_y,
            self.qk_y_plus,
            self.num_pieces,
            self.tol,
        )
        self.sk_x, self.delta_sk_x = ISQF.parametrize_spline(
            F,
            self.spline_knots,
            self.qk_x,
            self.qk_x_plus,
            self.num_pieces,
            self.tol,
        )

        if self.num_pieces > 1:
            self.sk_x_plus = F.concat(
                F.slice_axis(self.sk_x, axis=-1, begin=1, end=None),
                F.expand_dims(self.qk_x_plus, axis=-1),
                dim=-1,
            )
        else:
            self.sk_x_plus = F.expand_dims(self.qk_x_plus, axis=-1)

        # Get tails parameters
        self.tail_al, self.tail_bl = ISQF.parametrize_tail(
            F, self.beta_l, self.qk_x_l, self.qk_y_l
        )
        self.tail_ar, self.tail_br = ISQF.parametrize_tail(
            F, -self.beta_r, 1 - self.qk_x_r, self.qk_y_r
        )

    @property
    def F(self):
        return getF(self.beta_l)

    @property
    def args(self) -> List:
        return [
            self.spline_knots,
            self.spline_heights,
            self.beta_l,
            self.beta_r,
            self.qk_y_all,
        ]

    @staticmethod
    def parametrize_qk(
        F, quantile_knots: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Function to parametrize the x or y positions
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

        qk = F.slice_axis(quantile_knots, axis=-1, begin=0, end=-1)
        qk_plus = F.slice_axis(quantile_knots, axis=-1, begin=1, end=None)
        qk_l = F.slice_axis(quantile_knots, axis=-1, begin=0, end=1).squeeze(
            axis=-1
        )
        qk_r = F.slice_axis(
            quantile_knots, axis=-1, begin=-1, end=None
        ).squeeze(axis=-1)

        return qk, qk_plus, qk_l, qk_r

    @staticmethod
    def parametrize_spline(
        F,
        spline_knots: Tensor,
        qk: Tensor,
        qk_plus: Tensor,
        num_pieces: int,
        tol: float = 1e-4,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Function to parametrize the x or y positions of the spline knots
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

        # The spacing between spline knots is parametrized
        # by softmax function (in [0,1] and sum to 1)
        # We add tol to prevent overflow in computing 1/spacing in spline CRPS
        # After adding tol, it is normalized by
        # (1 + num_pieces * tol) to keep the sum-to-1 property
        delta_x = (F.softmax(spline_knots) + tol) / (1 + num_pieces * tol)

        # TODO: update to mxnet cumsum when it supports axis=-1
        x = cumsum(F, delta_x, exclusive=True)

        qk = F.expand_dims(qk, axis=-1)
        qk_plus = F.expand_dims(qk_plus, axis=-1)

        sk = F.broadcast_add(F.broadcast_mul(x, (qk_plus - qk)), qk)
        delta_sk = F.broadcast_mul(delta_x, (qk_plus - qk))

        return sk, delta_sk

    @staticmethod
    def parametrize_tail(
        F, beta: Tensor, qk_x: Tensor, qk_y: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Function to parametrize the tail parameters

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
        tail_b = -tail_a * F.log(qk_x) + qk_y

        return tail_a, tail_b

    def quantile(self, input_alpha: Tensor) -> Tensor:
        return self.quantile_internal(input_alpha, axis=0)

    def quantile_internal(
        self, alpha: Tensor, axis: Optional[int] = None
    ) -> Tensor:
        r"""
        Evaluates the quantile function at the quantile levels input_alpha
        Parameters
        ----------
        alpha
            Tensor of shape = (*batch_shape,) if axis=None, or containing an
            additional axis on the specified position, otherwise
        axis
            Index of the axis containing the different quantile levels which
            are to be computed.
            Read the description below for detailed information
        Returns
        -------
        Tensor
            Quantiles tensor, of the same shape as alpha
        """

        F = self.F
        qk_x, qk_x_l, qk_x_plus = self.qk_x, self.qk_x_l, self.qk_x_plus

        # The following describes the parameters reshaping in
        # quantile_internal, quantile_spline and quantile_tail

        # tail parameters: tail_al, tail_ar, tail_bl, tail_br,
        # shape = (*batch_shape,)
        # spline parameters: sk_x, sk_x_plus, sk_y, sk_y_plus,
        # shape = (*batch_shape, num_qk-1, num_pieces)
        # quantile knots parameters: qk_x, qk_x_plus, qk_y, qk_y_plus,
        # shape = (*batch_shape, num_qk-1)

        # axis=None - passed at inference when num_samples is None
        # shape of input_alpha = (*batch_shape,), will be expanded to
        # (*batch_shape, 1, 1) to perform operation
        # The shapes of parameters are as described above,
        # no reshaping is needed

        # axis=0 - passed at inference when num_samples is not None
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
        # We expand axis=0 for all of them

        # axis=-2 - passed at training when we evaluate quantiles at
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
        # We expand axis=-2 and axis=-1 for
        # spline and quantile knots parameters, respectively

        if axis is not None:
            qk_x_l = F.expand_dims(qk_x_l, axis=axis)
            qk_x = F.expand_dims(qk_x, axis=axis)
            qk_x_plus = F.expand_dims(qk_x_plus, axis=axis)

        quantile = F.where(
            F.broadcast_lesser(alpha, qk_x_l),
            self.quantile_tail(alpha, axis=axis, left_tail=True),
            self.quantile_tail(alpha, axis=axis, left_tail=False),
        )

        spline_val = self.quantile_spline(alpha, axis=axis)

        for spline_idx in range(self.num_qk - 1):

            is_in_between = F.broadcast_logical_and(
                F.broadcast_lesser_equal(
                    F.slice_axis(
                        qk_x, axis=-1, begin=spline_idx, end=spline_idx + 1
                    ).squeeze(-1),
                    alpha,
                ),
                F.broadcast_lesser(
                    alpha,
                    F.slice_axis(
                        qk_x_plus,
                        axis=-1,
                        begin=spline_idx,
                        end=spline_idx + 1,
                    ).squeeze(-1),
                ),
            )

            quantile = F.where(
                is_in_between,
                F.slice_axis(
                    spline_val, axis=-1, begin=spline_idx, end=spline_idx + 1
                ).squeeze(-1),
                quantile,
            )

        return quantile

    def quantile_spline(
        self,
        alpha: Tensor,
        axis: Optional[int] = None,
    ) -> Tensor:
        r"""
        Evaluates the spline functions at the
        quantile levels contained in alpha

        Parameters
        ----------
        alpha
            Input quantile levels
        axis
            Axis along which to expand
            For details of input_alpha shape and axis,
            refer to the description in quantile_internal
        Returns
        -------
        Tensor
            Quantiles tensor
            with shape
            = (*batch_shape, num_qk-1) if axis = None
            = (1, *batch_shape, num_qk-1) if axis = 0
            = (*batch_shape, num_qk-1, num_pieces) if axis = -2
        """

        F = self.F
        qk_y = self.qk_y
        sk_x, delta_sk_x, delta_sk_y = (
            self.sk_x,
            self.delta_sk_x,
            self.delta_sk_y,
        )

        if axis is not None:
            qk_y = F.expand_dims(qk_y, axis=0 if axis == 0 else -1)
            sk_x = F.expand_dims(sk_x, axis=axis)
            delta_sk_x = F.expand_dims(delta_sk_x, axis=axis)
            delta_sk_y = F.expand_dims(delta_sk_y, axis=axis)

        if axis is None or axis == 0:
            alpha = F.expand_dims(alpha, axis=-1)

        alpha = F.expand_dims(alpha, axis=-1)

        spline_val = F.broadcast_div(F.broadcast_sub(alpha, sk_x), delta_sk_x)
        spline_val = F.maximum(
            F.minimum(spline_val, F.ones_like(spline_val)),
            F.zeros_like(spline_val),
        )

        return F.broadcast_add(
            qk_y,
            F.sum(
                F.broadcast_mul(spline_val, delta_sk_y),
                axis=-1,
                keepdims=False,
            ),
        )

    def quantile_tail(
        self,
        alpha: Tensor,
        axis: Optional[int] = None,
        left_tail: bool = True,
    ) -> Tensor:
        r"""
        Evaluates the tail functions at the quantile levels contained in alpha
        Parameters
        ----------
        alpha
            Input quantile levels
        axis
            Axis along which to expand
            For details of input_alpha shape and axis,
            refer to the description in quantile_internal
        left_tail
            If True, compute the quantile for the left tail
            Otherwise, compute the quantile for the right tail
        Returns
        -------
        Tensor
            Quantiles tensor, of the same shape as alpha
        """

        F = self.F

        if left_tail:
            tail_a, tail_b = self.tail_al, self.tail_bl
        else:
            tail_a, tail_b = self.tail_ar, self.tail_br
            alpha = 1 - alpha

        if axis is not None:
            tail_a, tail_b = (
                F.expand_dims(tail_a, axis=axis),
                F.expand_dims(tail_b, axis=axis),
            )

        return F.broadcast_add(F.broadcast_mul(tail_a, F.log(alpha)), tail_b)

    def cdf_spline(self, z: Tensor) -> Tensor:
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

        F = self.F

        qk_y, qk_y_plus = self.qk_y, self.qk_y_plus
        qk_x, qk_x_plus = self.qk_x, self.qk_x_plus
        sk_x, delta_sk_x, delta_sk_y = (
            self.sk_x,
            self.delta_sk_x,
            self.delta_sk_y,
        )

        z_expand = F.expand_dims(z, axis=-1)

        if self.num_pieces > 1:
            qk_y_expand = F.expand_dims(qk_y, axis=-1)
            z_expand_twice = F.expand_dims(z_expand, axis=-1)

            knots_eval = self.quantile_spline(sk_x, axis=-2)

            # Compute \sum_{s=0}^{s_0-1} \Delta sk_y[s],
            # where \Delta sk_y[s] = (sk_y[s+1]-sk_y[s])
            mask_sum_s0 = F.broadcast_lesser(knots_eval, z_expand_twice)
            mask_sum_s0_minus = F.concat(
                F.slice_axis(mask_sum_s0, axis=-1, begin=1, end=None),
                F.zeros_like(qk_y_expand),
                dim=-1,
            )
            sum_delta_sk_y = F.sum(
                F.broadcast_mul(mask_sum_s0_minus, delta_sk_y),
                axis=-1,
                keepdims=False,
            )

            mask_s0_only = mask_sum_s0 - mask_sum_s0_minus
            # Compute (sk_x[s_0+1]-sk_x[s_0])/(sk_y[s_0+1]-sk_y[s_0])
            frac_s0 = F.sum(
                (mask_s0_only * delta_sk_x) / delta_sk_y,
                axis=-1,
                keepdims=False,
            )

            # Compute sk_x_{s_0}
            sk_x_s0 = F.sum(mask_s0_only * sk_x, axis=-1, keepdims=False)

            # Compute alpha_tilde
            alpha_tilde = (
                sk_x_s0
                + (F.broadcast_sub(z_expand, qk_y) - sum_delta_sk_y) * frac_s0
            )

        else:
            # num_pieces=1, ISQF reduces to IQF
            alpha_tilde = qk_x + F.broadcast_sub(z_expand, qk_y) / (
                qk_y_plus - qk_y
            ) * (qk_x_plus - qk_x)

        alpha_tilde = F.broadcast_minimum(
            F.broadcast_maximum(alpha_tilde, qk_x), qk_x_plus
        )

        return alpha_tilde

    def cdf_tail(self, z: Tensor, left_tail: bool = True) -> Tensor:
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

        F = self.F

        if left_tail:
            tail_a, tail_b, qk_x = self.tail_al, self.tail_bl, self.qk_x_l
        else:
            tail_a, tail_b, qk_x = self.tail_ar, self.tail_br, 1 - self.qk_x_r

        log_alpha_tilde = F.minimum((z - tail_b) / tail_a, F.log(qk_x))
        alpha_tilde = F.exp(log_alpha_tilde)
        return alpha_tilde if left_tail else 1 - alpha_tilde

    def crps_tail(self, z: Tensor, left_tail: bool = True) -> Tensor:
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

        F = self.F
        alpha_tilde = self.cdf_tail(z, left_tail=left_tail)

        if left_tail:
            tail_a, tail_b, qk_x, qk_y = (
                self.tail_al,
                self.tail_bl,
                self.qk_x_l,
                self.qk_y_l,
            )
            term1 = (z - tail_b) * (qk_x**2 - 2 * qk_x + 2 * alpha_tilde)
            term2 = qk_x**2 * tail_a * (-F.log(qk_x) + 0.5)
            term2 = term2 + 2 * F.where(
                z < qk_y,
                qk_x * tail_a * (F.log(qk_x) - 1)
                + alpha_tilde * (-z + tail_b + tail_a),
                F.zeros_like(qk_x),
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
                + (qk_x**2 - 1) * F.log(1 - qk_x)
                + 2 * alpha_tilde
            )
            term2 = term2 + 2 * F.where(
                z > qk_y,
                (1 - alpha_tilde) * (z - tail_b),
                tail_a * (1 - qk_x) * F.log(1 - qk_x),
            )

        return term1 + term2

    def crps_spline(self, z: Tensor) -> Tensor:
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

        F = self.F
        qk_x, qk_x_plus, qk_y = self.qk_x, self.qk_x_plus, self.qk_y
        sk_x, sk_x_plus = self.sk_x, self.sk_x_plus
        delta_sk_x, delta_sk_y = self.delta_sk_x, self.delta_sk_y

        z_expand, qk_x_plus_expand = (
            F.expand_dims(z, axis=-1),
            F.expand_dims(qk_x_plus, axis=-1),
        )

        alpha_tilde = self.cdf_spline(z)
        alpha_tilde_expand = F.expand_dims(alpha_tilde, axis=-1)

        r = F.broadcast_minimum(
            F.broadcast_maximum(alpha_tilde_expand, sk_x), sk_x_plus
        )

        coeff1 = (
            -2 / 3 * sk_x_plus**3
            + sk_x * sk_x_plus**2
            + sk_x_plus**2
            - (1 / 3) * sk_x**3
            - 2 * sk_x * sk_x_plus
            - r**2
            + 2 * sk_x * r
        )

        coeff2 = F.broadcast_add(
            -2 * F.broadcast_maximum(alpha_tilde_expand, sk_x_plus)
            + sk_x_plus**2,
            2 * qk_x_plus_expand - qk_x_plus_expand**2,
        )

        result = (
            (qk_x_plus**2 - qk_x**2) * F.broadcast_sub(z_expand, qk_y)
            + 2
            * F.broadcast_sub(qk_x_plus, alpha_tilde)
            * F.broadcast_sub(qk_y, z_expand)
            + F.sum(
                (delta_sk_y / delta_sk_x) * coeff1, axis=-1, keepdims=False
            )
            + F.sum(delta_sk_y * coeff2, axis=-1, keepdims=False)
        )

        return F.sum(result, axis=-1, keepdims=False)

    def loss(self, z: Tensor) -> Tensor:
        return self.crps(z)

    def crps(self, z: Tensor) -> Tensor:
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

    def cdf(self, z: Tensor) -> Tensor:
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

        F = self.F
        qk_y, qk_y_l, qk_y_plus = self.qk_y, self.qk_y_l, self.qk_y_plus

        alpha_tilde = F.where(
            z < qk_y_l,
            self.cdf_tail(z, left_tail=True),
            self.cdf_tail(z, left_tail=False),
        )

        spline_alpha_tilde = self.cdf_spline(z)

        for i in range(self.num_qk - 1):

            is_in_between = F.broadcast_logical_and(
                F.slice_axis(qk_y, axis=-1, begin=i, end=i + 1).squeeze(-1)
                <= z,
                z
                < F.slice_axis(qk_y_plus, axis=-1, begin=i, end=i + 1).squeeze(
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

        F = self.F

        # if num_samples=None then input_alpha should have the same shape
        # as beta_l, i.e., (*batch_shape,)
        # else u should be (num_samples, *batch_shape)
        alpha = F.random.uniform_like(
            data=(
                self.beta_l
                if num_samples is None
                else self.beta_l.expand_dims(axis=0).repeat(
                    axis=0, repeats=num_samples
                )
            )
        )

        sample = self.quantile(alpha)

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
    r"""
    DistributionOutput class for the Incremental (Spline) Quantile Function

    Parameters
    ----------
    num_pieces
        number of spline pieces for each spline
        ISQF reduces to IQF when num_pieces = 1
    alpha
        Tensor containing the x-positions of quantile knots
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
        F,
        *args: Tensor,
        tol: float = 1e-4,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Domain map function The inputs of this function are specified by
        self.args_dim knots, heights:

        parameterizing the x-/ y-positions of the spline knots,
        shape = (*batch_shape, (num_qk-1)*num_pieces)
        q:
        parameterizing the y-positions of the quantile knots,
        shape = (*batch_shape, num_qk)
        beta_l, beta_r:
        parameterizing the left/right tail, shape = (*batch_shape, 1)
        """

        try:
            spline_knots, spline_heights, beta_l, beta_r, quantile_knots = args
        except ValueError:
            raise ValueError(
                "Failed to unpack args of domain_map. Double check your input."
            )

        # Add tol to prevent the y-distance of
        # two quantile knots from being too small
        #
        # Because in this case the spline knots could be squeezed together
        # and cause overflow in spline CRPS computation
        qk_y = F.concat(
            F.slice_axis(quantile_knots, axis=-1, begin=0, end=1),
            F.abs(F.slice_axis(quantile_knots, axis=-1, begin=1, end=None))
            + tol,
            dim=-1,
        )
        # TODO: update to mxnet cumsum when it supports axis=-1
        qk_y = cumsum(F, qk_y)

        # Prevent overflow when we compute 1/beta
        beta_l, beta_r = (
            F.abs(beta_l.squeeze(axis=-1)) + tol,
            F.abs(beta_r.squeeze(axis=-1)) + tol,
        )

        return spline_knots, spline_heights, beta_l, beta_r, qk_y

    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> ISQF:
        """
        function outputing the distribution class
        distr_args: distribution arguments
        loc: shift to the data mean
        scale: scale to the data
        """

        distr_args, qk_x = self.reshape_spline_args(distr_args, self.qk_x)

        if scale is None:
            return self.distr_cls(
                *distr_args, qk_x, self.num_qk, self.num_pieces, self.tol
            )
        else:
            distr = self.distr_cls(
                *distr_args, qk_x, self.num_qk, self.num_pieces, self.tol
            )
            return TransformedISQF(
                distr, [AffineTransformation(loc=loc, scale=scale)]
            )

    def reshape_spline_args(self, distr_args, qk_x):
        """
        auxiliary function reshaping knots and heights to (*batch_shape,
        num_qk-1, num_pieces) alpha to (*batch_shape, num_qk)
        """

        spline_knots, spline_heights = distr_args[0], distr_args[1]
        beta_l = distr_args[2]
        qk_y = distr_args[4]
        F = getF(beta_l)

        # FIXME number 1
        # Convert alpha from list of len=num_qk to
        # Tensor of shape (*batch_shape, num_qk)
        #
        # For example, if alpha = [0.1, 0.5, 0.9],
        # then alpha_reshape will be a Tensor of shape (*batch_shape, 3)
        # with the last dimension being [0.1, 0.5, 0.9]
        #
        # In PyTorch, it would be torch.tensor(alpha).repeat(*batch_shape,1)
        qk_x_reshape = F.concat(
            *[
                F.expand_dims(F.ones_like(beta_l), axis=-1) * qk_x[i]
                for i in range(self.num_qk)
            ],
            dim=-1,
        )

        # FIXME number 2
        # knots and heights have shape (*batch_shape, (num_qk-1)*num_pieces)
        # I want to convert the shape to (*batch_shape, (num_qk-1), num_pieces)
        # Here I make a shape_holder with target_shape, and use reshape_like

        # create a shape holder of shape (*batch_shape, num_qk-1, num_pieces)
        shape_holder = F.repeat(
            F.expand_dims(
                F.slice_axis(qk_y, axis=-1, begin=0, end=-1), axis=-1
            ),
            repeats=self.num_pieces,
            axis=-1,
        )

        spline_knots_reshape = F.reshape_like(spline_knots, shape_holder)
        spline_heights_reshape = F.reshape_like(spline_heights, shape_holder)
        distr_args_reshape = (
            spline_knots_reshape,
            spline_heights_reshape,
        ) + distr_args[2:]

        return distr_args_reshape, qk_x_reshape

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedISQF(TransformedDistribution, ISQF):
    # DistributionOutput class for the case when loc/scale is not None
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

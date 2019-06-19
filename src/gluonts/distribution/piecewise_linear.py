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

# Standard library imports
from typing import Dict, Optional, Tuple, cast

# First-party imports
from gluonts.core.component import validated
from gluonts.distribution.bijection import AffineTransformation, Bijection
from gluonts.distribution.distribution import Distribution, getF
from gluonts.distribution.distribution_output import DistributionOutput
from gluonts.distribution.transformed_distribution import (
    TransformedDistribution,
)
from gluonts.model.common import Tensor
from gluonts.support import util


class PiecewiseLinear(Distribution):
    r"""
    Piecewise linear distribution.

    This class represents the *quantile function* (i.e., the inverse CDF)
    associated with the a distribution, as a continuous, non-decreasing,
    piecewise linear function defined in the [0, 1] interval:

    .. math::
        q(x; \gamma, b, d) = \gamma + \sum_{l=0}^L b_l (x_l - d_l)_+

    where the input :math:`x \in [0,1]` and the parameters are

    - :math:`\gamma`: intercept at 0
    - :math:`b`: differences of the slopes in consecutive pieces
    - :math:`d`: knot positions

    Parameters
    ----------
    gamma
        Tensor containing the intercepts at zero
    slopes
        Tensor containing the slopes of each linear piece.
        All coefficients must be positive.
        Shape: (*gamma.shape, num_pieces)
    knot_spacings
        Tensor containing the spacings between knots in the splines.
        All coefficients must be positive and sum to one on the last axis.
        Shape: (*gamma.shape, num_pieces)
    F
    """

    is_reparameterizable = False

    def __init__(
        self, gamma: Tensor, slopes: Tensor, knot_spacings: Tensor, F=None
    ) -> None:
        self.F = F if F else getF(gamma)
        self.gamma = gamma

        # Since most of the calculations are easily expressed in the original parameters, we transform the
        # learned parameters back
        self.b, self.knot_positions = PiecewiseLinear._to_orig_params(
            self.F, slopes, knot_spacings
        )

    @staticmethod
    def _to_orig_params(
        F, slopes: Tensor, knot_spacings: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Convert the trainable parameters to the original parameters of the
        splines, i.e., convert the slopes of each piece to the difference
        between slopes of consecutive pieces and knot spacings to knot
        positions.

        Parameters
        ----------
        F
        slopes
            Tensor of shape (*gamma.shape, num_pieces)
        knot_spacings
            Tensor of shape (*gamma.shape, num_pieces)

        Returns
        -------
        Tensor
            Tensor of shape (*gamma.shape, num_pieces)
        Tensor
            Tensor of shape (*gamma.shape, num_pieces)
        """

        # b: the difference between slopes of consecutive pieces
        # shape (..., num_pieces - 1)
        b = F.slice_axis(slopes, axis=-1, begin=1, end=None) - F.slice_axis(
            slopes, axis=-1, begin=0, end=-1
        )

        # Add slope of first piece to b: b_0 = m_0
        m_0 = F.slice_axis(slopes, axis=-1, begin=0, end=1)
        b = F.concat(m_0, b, dim=-1)

        # The actual position of the knots is obtained by cumulative sum of
        # the knot spacings. The first knot position is always 0 for quantile
        # functions; cumsum will take care of that.
        knot_positions = util.cumsum(F, knot_spacings, exclusive=True)

        return b, knot_positions

    def sample(self, num_samples: Optional[int] = None) -> Tensor:
        F = self.F

        # if num_samples=None then u should have the same shape as gamma, i.e., (dim,)
        # else u should be (num_samples, dim)
        # Note: there is no need to extend the parameters to (num_samples, dim, ...)
        u = F.random.uniform_like(
            data=(
                self.gamma
                if num_samples is None
                else self.gamma.expand_dims(axis=0).repeat(
                    axis=0, repeats=num_samples
                )
            )
        )

        sample = self.quantile(u, axis=None if num_samples is None else 0)

        return sample

    # overwrites the loss method of the Distribution class
    def loss(self, x: Tensor) -> Tensor:
        return self.crps(x)

    def crps(self, x: Tensor) -> Tensor:
        r"""
        Compute CRPS in analytical form.

        Parameters
        ----------
        x
            Observation to evaluate. Shape equals to gamma.shape.

        Returns
        -------
        Tensor
            Tensor containing the CRPS.
        """

        F = self.F
        gamma, b, knot_positions = self.gamma, self.b, self.knot_positions

        a_tilde = self.cdf(x)

        max_a_tilde_knots = F.broadcast_maximum(
            a_tilde.expand_dims(axis=-1), knot_positions
        )

        knots_cubed = F.broadcast_power(self.knot_positions, F.ones(1) * 3.0)

        coeff = (
            (1.0 - knots_cubed) / 3.0
            - knot_positions
            - F.square(max_a_tilde_knots)
            + 2 * max_a_tilde_knots * knot_positions
        )

        crps = (
            (2 * a_tilde - 1) * x
            + (1 - 2 * a_tilde) * gamma
            + F.sum(b * coeff, axis=-1, keepdims=False)
        )

        return crps

    def cdf(self, x: Tensor) -> Tensor:
        r"""
        Computes the quantile level :math:`\alpha` such that
        :math:`q(\alpha) = x`.

        Parameters
        ----------
        x
            Tensor of shape gamma.shape

        Returns
        -------
        Tensor
            Tensor of shape gamma.shape
        """

        F = self.F
        gamma, b, knot_positions = self.gamma, self.b, self.knot_positions

        quantiles_at_knots = self.quantile(knot_positions, axis=-2)

        # Mask to nullify the terms corresponding to knots larger than l_0, which is the largest knot
        # (quantile level) such that the quantile at l_0, s(l_0) < x.
        # (..., num_pieces)
        mask = F.broadcast_lesser(quantiles_at_knots, x.expand_dims(axis=-1))

        slope_l0 = F.sum(b * mask, axis=-1, keepdims=False)

        # slope_l0 can be zero in which case a_tilde = 0.
        # The following is to circumvent mxnet issue with "where" operator which returns nan even if the statement
        # you are interested in does not result in nan (but the "else" statement evaluates to nan).
        slope_l0_nz = F.where(
            slope_l0 == F.zeros_like(slope_l0), F.ones_like(x), slope_l0
        )

        a_tilde = F.where(
            slope_l0 == F.zeros_like(slope_l0),
            F.zeros_like(x),
            (
                x
                - gamma
                + F.sum(b * knot_positions * mask, axis=-1, keepdims=False)
            )
            / slope_l0_nz,
        )

        return a_tilde

    def quantile(self, x: Tensor, axis: Optional[int] = None) -> Tensor:
        r"""
        Evaluates the quantile function at the quantile levels contained in `x`.

        Parameters
        ----------
        x
            Tensor of shape *gamma.shape if axis=None, or containing an
            additional axis on the specified position, otherwise.
        axis
            Index of the axis containing the different quantile levels which
            are to be computed.

        Returns
        -------
        Tensor
            Quantiles tensor, of the same shape as x.
        """

        F = self.F

        # shapes of self
        # self.gamma: (*batch_shape)
        # self.knot_positions, self.b: (*batch_shape, num_pieces)

        # axis=None - passed at inference when num_samples is None
        # The shape of x is (*batch_shape).
        # The shapes of the parameters should be:
        # gamma: (*batch_shape), knot_positions, b: (*batch_shape, num_pieces)
        # They match the self. counterparts so no reshaping is needed

        # axis=0 - passed at inference when num_samples is not None
        # The shape of x is (num_samples, *batch_shape).
        # The shapes of the parameters should be:
        # gamma: (num_samples, *batch_shape), knot_positions, b: (num_samples, *batch_shape, num_pieces),
        # They do not match the self. counterparts and we need to expand the axis=0 to all of them.

        # axis=-2 - passed at training when we evaluate quantiles at knot_positions in order to compute a_tilde
        # The shape of x is shape(x) = shape(knot_positions) = (*batch_shape, num_pieces).
        # The shape of the parameters shopuld be:
        # gamma: (*batch_shape, 1), knot_positions: (*batch_shape, 1, num_pieces), b: (*batch_shape, 1, num_pieces)
        # They do not match the self. counterparts and we need to expand axis=-1 for gamma and axis=-2 for the rest.

        if axis is not None:
            gamma = self.gamma.expand_dims(axis=axis if axis == 0 else -1)
            knot_positions = self.knot_positions.expand_dims(axis=axis)
            b = self.b.expand_dims(axis=axis)
        else:
            gamma, knot_positions, b = self.gamma, self.knot_positions, self.b

        x_minus_knots = F.broadcast_minus(
            x.expand_dims(axis=-1), knot_positions
        )

        quantile = F.broadcast_add(
            gamma, F.sum(F.broadcast_mul(b, F.relu(x_minus_knots)), axis=-1)
        )

        return quantile

    @property
    def batch_shape(self) -> Tuple:
        return self.gamma.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0


class PiecewiseLinearOutput(DistributionOutput):
    distr_cls: type = PiecewiseLinear

    @validated()
    def __init__(self, num_pieces: int) -> None:
        assert (
            isinstance(num_pieces, int) and num_pieces > 1
        ), "num_pieces should be an integer larger than 1"

        self.num_pieces = num_pieces
        self.args_dim = cast(
            Dict[str, int],
            {"gamma": 1, "slopes": num_pieces, "knot_spacings": num_pieces},
        )

    @classmethod
    def domain_map(cls, F, gamma, slopes, knot_spacings):
        # slopes of the pieces are non-negative
        thresh = F.zeros_like(slopes) + 20.0
        I_grt_thresh = F.broadcast_greater_equal(slopes, thresh)
        I_sml_thresh = F.broadcast_lesser(slopes, thresh)
        slopes_proj = (
            I_sml_thresh
            * F.Activation(data=(I_sml_thresh * slopes), act_type='softrelu')
            + I_grt_thresh * slopes
        )
        # slopes = F.Activation(slopes, 'softrelu')

        # the spacing between the knots should be in [0, 1] and sum to 1
        knot_spacings_proj = F.softmax(knot_spacings)

        return gamma.squeeze(axis=-1), slopes_proj, knot_spacings_proj

    def distribution(
        self, distr_args, scale: Optional[Tensor] = None, **kwargs
    ) -> PiecewiseLinear:
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedPiecewiseLinear(
                distr, AffineTransformation(scale=scale)
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


# Need to inherit from PiecewiseLinear to get the overwritten loss method.
class TransformedPiecewiseLinear(TransformedDistribution, PiecewiseLinear):
    def __init__(
        self, base_distribution: PiecewiseLinear, *transforms: Bijection
    ) -> None:
        super().__init__(base_distribution, *transforms)

    def crps(self, y: Tensor) -> Tensor:
        # TODO: use event_shape
        F = getF(y)

        for t in self.transforms[::-1]:
            assert isinstance(
                t, AffineTransformation
            ), "Not an AffineTransformation"
            assert (
                t.scale is not None and t.loc is None
            ), "Not a scaling transformation"

            scale = t.scale
            x = t.f_inv(y)

        # (..., 1)
        p = self.base_distribution.crps(x)

        return F.broadcast_mul(p, scale)

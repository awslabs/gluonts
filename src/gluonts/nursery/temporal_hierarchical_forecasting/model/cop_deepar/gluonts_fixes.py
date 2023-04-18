from functools import partial
from typing import Callable, Iterator, List, Optional, cast, Dict, Tuple
from pathlib import Path

import mxnet as mx
from gluonts.mx.distribution.bijection import AffineTransformation
from mxnet import gluon
import numpy as np


from gluonts.core.component import Type, validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import DataBatch, InferenceDataLoader
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.deepar._network import DeepARPredictionNetwork
from gluonts.core.serde import load_json
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    SampleForecastGenerator,
)
from gluonts.mx import Tensor
from gluonts.mx.batchify import stack
from gluonts.mx.distribution import DistributionOutput, EmpiricalDistribution, PiecewiseLinear
from gluonts.mx.distribution.distribution_output import ArgProj
from gluonts.mx.distribution.piecewise_linear import PiecewiseLinearOutput, TransformedPiecewiseLinear
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.transform import Transformation
from gluonts.mx.util import (
    export_repr_block,
    export_symb_block,
    get_hybrid_forward_input_names,
    hybrid_block_to_symbol_block,
    import_repr_block,
    import_symb_block,
)
from gluonts.mx.context import get_mxnet_context

LOSS_FUNCTIONS = ["crps_univariate", "mse", "nll"]


def batchify_with_dict(
    data: List[dict],
    ctx: Optional[mx.context.Context] = None,
    dtype: Optional[Type] = np.float32,
    variable_length: bool = False,
    is_right_pad: bool = True,
) -> DataBatch:
    return {
        key: stack(
            data=[item[key] for item in data],
            ctx=ctx,
            dtype=dtype,
            variable_length=variable_length,
            is_right_pad=is_right_pad,
        ) if not isinstance(data[0][key], dict)
        else batchify_with_dict(data=[item[key] for item in data])
        for key in data[0].keys()
    }


class RepresentableBlockPredictorBatchifyWithDict(RepresentableBlockPredictor):
    """
    We need the stack function `batchify_with_dict` in order to pass the features at the aggregated level properly
    during prediction. Gluonts does not allow this without changing the line corresponding to the
    `InferenceDataLoader` in the `predict` function.
    """

    BlockType = mx.gluon.HybridBlock

    def __init__(
        self,
        prediction_net: BlockType,
        batch_size: int,
        prediction_length: int,
        freq: str,
        ctx: mx.Context,
        input_transform: Transformation,
        lead_time: int = 0,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[
            Callable[[DataEntry, np.ndarray], np.ndarray]
        ] = None,
        dtype: Type = np.float32,
    ) -> None:
        super().__init__(
            prediction_net=prediction_net,
            batch_size=batch_size,
            prediction_length=prediction_length,
            ctx=ctx,
            input_transform=input_transform,
            lead_time=lead_time,
            forecast_generator=forecast_generator,
            output_transform=output_transform,
            dtype=dtype,
        )

    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=partial(batchify_with_dict, ctx=self.ctx, dtype=self.dtype),
        )
        with mx.Context(self.ctx):
            yield from self.forecast_generator(
                inference_data_loader=inference_data_loader,
                prediction_net=self.prediction_net,
                input_names=self.input_names,
                output_transform=self.output_transform,
                num_samples=num_samples,
            )

    @classmethod
    def deserialize(
            cls, path: Path, ctx: Optional[mx.Context] = None
    ) -> "RepresentableBlockPredictorBatchifyWithDict":
        ctx = ctx if ctx is not None else get_mxnet_context()

        with mx.Context(ctx):
            # deserialize constructor parameters
            with (path / "parameters.json").open("r") as fp:
                parameters = load_json(fp.read())

            # deserialize transformation chain
            with (path / "input_transform.json").open("r") as fp:
                transform = load_json(fp.read())

            # deserialize prediction network
            prediction_net = import_repr_block(path, "prediction_net")

            # input_names is derived from the prediction_net
            if "input_names" in parameters:
                del parameters["input_names"]

            parameters["ctx"] = ctx
            return RepresentableBlockPredictorBatchifyWithDict(
                input_transform=transform,
                prediction_net=prediction_net,
                batch_size=parameters['batch_size'],
                freq=parameters['freq'],
                prediction_length=parameters['prediction_length'],
                ctx=parameters['ctx'],
                dtype=parameters['dtype']
            )

# Gluonts estimator should expose this function. Currently it has api for creating predictor but not prediction network!
def create_prediction_network(
    estimator: DeepAREstimator,
) -> DeepARPredictionNetwork:
    return DeepARPredictionNetwork(
        num_parallel_samples=estimator.num_parallel_samples,
        num_layers=estimator.num_layers,
        num_cells=estimator.num_cells,
        cell_type=estimator.cell_type,
        history_length=estimator.history_length,
        context_length=estimator.context_length,
        prediction_length=estimator.prediction_length,
        distr_output=estimator.distr_output,
        dropoutcell_type=estimator.dropoutcell_type,
        dropout_rate=estimator.dropout_rate,
        cardinality=estimator.cardinality,
        embedding_dimension=estimator.embedding_dimension,
        lags_seq=estimator.lags_seq,
        scaling=estimator.scaling,
        dtype=estimator.dtype,
        num_imputation_samples=estimator.num_imputation_samples,
        default_scale=estimator.default_scale,
        minimum_scale=estimator.minimum_scale,
        impute_missing_values=estimator.impute_missing_values,
    )


class ArgProjFixed(ArgProj):
    def __init__(
        self,
        args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[Tensor]],
        dtype: Type = np.float32,
        prefix: Optional[str] = None,
        non_negative: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            args_dim=args_dim,
            domain_map=domain_map,
            dtype=dtype,
            prefix=prefix,
            **kwargs,
        )
        self.non_negative = non_negative

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, x: Tensor, **kwargs) -> Tuple[Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        # change: passed along `kwargs`
        return self.domain_map(*params_unbounded, self.non_negative)


class PiecewiseLinearOutputFixed(PiecewiseLinearOutput):

    def get_args_proj(self, non_negative: bool = False, prefix: Optional[str] = None) -> gluon.HybridBlock:
        # change: call our own version of `ArgProj`.
        return ArgProjFixed(
            args_dim=self.args_dim,
            domain_map=gluon.nn.HybridLambda(self.domain_map),
            prefix=prefix,
            dtype=self.dtype,
            non_negative=non_negative,
        )

    @classmethod
    def domain_map(cls, F, gamma, slopes, knot_spacings, non_negative: bool = False):
        # slopes of the pieces are non-negative
        slopes_proj = F.Activation(data=slopes, act_type="softrelu") + 1e-4

        # the spacing between the knots should be in [0, 1] and sum to 1
        knot_spacings_proj = F.softmax(knot_spacings)

        # change: non-negative check.
        if non_negative:
            gamma = F.Activation(data=gamma, act_type="softrelu")

        return gamma.squeeze(axis=-1), slopes_proj, knot_spacings_proj


class EmpiricalDistributionWithPointMetrics(EmpiricalDistribution):

    def mse(self, x: Tensor) -> Tensor:
        r"""
        Compute the mean squared error (MSE) of `x` according to the empirical distribution.

        The last dimension of `x` specifies the "event dimension" of the target (= 1 for the univariate case).
        For multivariate target, MSE scores are computed for each dimension separately and then their sum is returned.

        Parameters
        ----------
        x
            Tensor of ground truth with shape `(*batch_shape, *event_shape)`

        Returns
        -------
        Tensor
            MSE of shape `(*batch_shape, 1)`.
        """
        F = self.F
        mean_forecast = self.mean

        # mse shape: (*batch_shape, *event_shape)
        mse = F.square(mean_forecast - x) / np.prod(x.shape)

        # Sum the axis corresponding to the target (event) dimension.
        if self.event_dim > 0:
            # Total MSE: sum over all but the axes corresponding to the batch shape.
            # Shape: `(*batch_shape)`
            mse = F.sum(
                mse, exclude=True, axis=list(range(0, len(self.batch_shape)))
            )

        return mse

    def loss(self, x: Tensor, loss_function:str = "crps_univariate") -> Tensor:
        assert loss_function in LOSS_FUNCTIONS

        if loss_function in "crps_univariate":
            return self.crps_univariate(x=x)
        else:
            return self.mse(x=x)


# Gluonts `PiecewiseLinear` should have this implemented already.
class PiecewiseLinearWithSampling(PiecewiseLinear):
    def sample_rep(
        self, num_samples: Optional[int] = None, dtype=float
    ) -> Tensor:
        return self.sample(num_samples=num_samples, dtype=dtype)


class PiecewiseLinearVector(PiecewiseLinearWithSampling):
    r"""
    Piecewise linear distribution.

    This class represents d-independent *quantile functions* (i.e., the inverse CDFs)
    associated with some underlying distributions, as a continuous, non-decreasing,
    piecewise linear functions defined in the [0, 1] interval:

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
        Shape: ``(*gamma.shape, num_pieces)``
    knot_spacings
        Tensor containing the spacings between knots in the splines.
        All coefficients must be positive and sum to one on the last axis.
        Shape: ``(*gamma.shape, num_pieces)``
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(
        self, gamma: Tensor, slopes: Tensor, knot_spacings: Tensor
    ) -> None:
        self.gamma = gamma
        self.slopes = slopes.reshape(gamma.shape + (-1,))
        self.knot_spacings = knot_spacings.reshape(gamma.shape + (-1,))

        # Since most of the calculations are easily expressed in the original parameters, we transform the
        # learned parameters back
        self.b, self.knot_positions = PiecewiseLinear._to_orig_params(
            self.F, self.slopes, self.knot_spacings
        )

    def event_shape(self) -> Tuple:
        return self.gamma.shape[-1:]

    @property
    def event_dim(self) -> int:
        return 1


class PiecewiseLinearVectorOutput(DistributionOutput):
    distr_cls: type = PiecewiseLinearVector

    @validated()
    def __init__(self, num_pieces: int, dim: int) -> None:
        super().__init__(self)
        self.dim = dim

        assert (
            isinstance(num_pieces, int) and num_pieces > 1
        ), "num_pieces should be an integer larger than 1"

        self.num_pieces = num_pieces
        self.args_dim = cast(
            Dict[str, int],
            {"gamma": dim, "slopes": dim * num_pieces, "knot_spacings": dim * num_pieces},
        )

    @classmethod
    def domain_map(cls, F, gamma, slopes, knot_spacings, non_negative: bool = False):
        # slopes of the pieces are non-negative
        slopes_proj = F.Activation(data=slopes, act_type="softrelu") + 1e-4

        # the spacing between the knots should be in [0, 1] and sum to 1
        knot_spacings_proj = F.softmax(knot_spacings)

        # change: non-negative check.
        if non_negative:
            gamma = F.Activation(data=gamma, act_type="softrelu")

        return gamma, slopes_proj, knot_spacings_proj

    def get_args_proj(self, non_negative: bool = False, prefix: Optional[str] = None) -> gluon.HybridBlock:
        # change: call our own version of `ArgProj`.
        return ArgProjFixed(
            args_dim=self.args_dim,
            domain_map=gluon.nn.HybridLambda(self.domain_map),
            prefix=prefix,
            dtype=self.dtype,
            non_negative=non_negative,
        )

    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> PiecewiseLinear:
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedPiecewiseLinear(
                distr, [AffineTransformation(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return (self.dim, )
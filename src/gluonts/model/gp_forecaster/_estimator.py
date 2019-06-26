# Standard library imports
from typing import List, Optional

# Third-party imports
import numpy as np
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts import transform
from gluonts.core.component import DType, validated
from gluonts.kernels import KernelOutput, RBFKernelOutput
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
from gluonts.time_feature.lag import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.trainer import Trainer
from gluonts.transform import (
    AddTimeFeatures,
    AsNumpyArray,
    CanonicalInstanceSplitter,
    Chain,
    FieldName,
    SetFieldIfNotPresent,
    TestSplitSampler,
    Transformation,
)

# Relative imports
from ._network import (
    GaussianProcessPredictionNetwork,
    GaussianProcessTrainingNetwork,
)


class GaussianProcessEstimator(GluonEstimator):
    r"""
    GaussianProcessEstimator shows how to build a local time series model using
    Gaussian Processes (GP).

    Each time series has a GP with its own
    hyper-parameters.  For the radial basis function (RBF) Kernel, the
    learnable hyper-parameters are the amplitude and lengthscale. The periodic
    kernel has those hyper-parameters with an additional learnable frequency
    parameter. The RBFKernel is the default, but either kernel can be used by
    inputting the desired KernelOutput object. The noise sigma in the model is
    another learnable hyper-parameter for both kernels. These parameters are
    fit using an Embedding of the integer time series indices (each time series
    has its set of hyper-parameter that is static in time). The observations
    are the time series values. In this model, the time features are hour of
    the day and day of the week.

    Parameters
    ----------
    freq
        Time series frequency.
    prediction_length
        Prediction length.
    cardinality
        Number of time series.
    trainer
        Trainer instance to be used for model training.
    context_length
        Training length.
    num_eval_samples
        Number of samples to be drawn.
    kernel_output
        KernelOutput instance to determine which kernel subclass to be
        instantiated.
    params_scaling
        Determines whether or not to scale the model parameters.
    float_type
        Determines whether to use single or double precision.
    max_iter_jitter
        Maximum number of iterations for jitter to iteratively make the matrix positive definite.
    jitter_method
        Iteratively jitter method or use eigenvalue decomposition depending on problem size.
    sample_noise
        Boolean to determine whether to add :math:`\sigma^2I` to the predictive covariance matrix.
    time_features
        Time features to use as inputs of the model (default: None, in which
        case these are automatically determined based on the frequency.)
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_eval_samples: int = 100,
        kernel_output: KernelOutput = RBFKernelOutput(),
        params_scaling: bool = True,
        float_type: DType = np.float64,
        max_iter_jitter: int = 10,
        jitter_method: str = 'iter',
        sample_noise: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
    ) -> None:
        self.float_type = float_type
        super().__init__(trainer=trainer, float_type=self.float_type)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert cardinality > 0, "The value of `cardinality` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert (
            num_eval_samples > 0
        ), "The value of `num_eval_samples` should be > 0"

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.cardinality = cardinality
        self.num_eval_samples = num_eval_samples
        self.kernel_output = kernel_output
        self.params_scaling = params_scaling
        self.max_iter_jitter = max_iter_jitter
        self.jitter_method = jitter_method
        self.sample_noise = sample_noise
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                AsNumpyArray(
                    field=transform.FieldName.FEAT_STATIC_CAT, expected_ndim=1
                ),
                CanonicalInstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    instance_sampler=TestSplitSampler(),
                    time_series_fields=[FieldName.FEAT_TIME],
                    instance_length=self.context_length,
                    use_prediction_features=True,
                    prediction_length=self.prediction_length,
                ),
            ]
        )

    def create_training_network(self) -> HybridBlock:
        return GaussianProcessTrainingNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            cardinality=self.cardinality,
            kernel_output=self.kernel_output,
            params_scaling=self.params_scaling,
            ctx=self.trainer.ctx,
            float_type=self.float_type,
            max_iter_jitter=self.max_iter_jitter,
            jitter_method=self.jitter_method,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = GaussianProcessPredictionNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            cardinality=self.cardinality,
            num_samples=self.num_eval_samples,
            params=trained_network.collect_params(),
            kernel_output=self.kernel_output,
            params_scaling=self.params_scaling,
            ctx=self.trainer.ctx,
            float_type=self.float_type,
            max_iter_jitter=self.max_iter_jitter,
            jitter_method=self.jitter_method,
            sample_noise=self.sample_noise,
        )

        copy_parameters(
            net_source=trained_network, net_dest=prediction_network
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            float_type=self.float_type,
        )

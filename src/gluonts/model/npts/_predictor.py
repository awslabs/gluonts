# Standard library imports
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import validated
from gluonts.core.exception import GluonTSDataError
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature.lag import time_features_from_frequency_str

# Relative imports
from ._model import NPTS


class KernelType(str, Enum):
    exponential = 'exponential'
    uniform = 'uniform'


class NPTSPredictor(RepresentablePredictor):
    r"""
    Implementation of Non-Parametric Time Series Forecaster.

    Forecasts of NPTS for time step :math:`T` are one of the previous values
    of the time series (these could be known values or predictions), sampled
    according to the (un-normalized) distribution :math:`q_T(t) > 0`, where
    :math:`0 <= t < T`.

    The distribution :math:`q_T` is expressed in terms of a feature map
    :math:`f(t)` which associates a time step :math:`t` with a
    :math:`D`-dimensional feature map :math:`[f_1(t), ..., f_D(t)]`. More
    details on the feature map can be found below.

    We offer two types of distribution kernels.

    **Exponential Kernel (NPTS Forecaster)**

      The sampling distribution :math:`q_T` for the `exponential` kernel
      can be `weighted` or `unweighted` and is defined as follows.

      .. math::

        q_T(t) =
        \begin{cases}
          \exp( - \sum_{i=1}^D \alpha   \left| f_i(t) - f_i(T) \right| )
            & \text{unweighted}\\
          \exp( - \sum_{i=1}^D \alpha_i \left| f_i(t) - f_i(T) \right| )
            & \text{weighted}
        \end{cases}

      In the above definition :math:`\alpha > 0` and :math:`\alpha_i > 0` are
      user-defined sampling weights.

    **Uniform Kernel (Climatological Forecaster)**

      The sampling distribution :math:`q_T` for the `uniform` kernel can be
      `seasonal` or not. The `seasonal` version is defined as follows.

      .. math::

         q_T(t) =
         \begin{cases}
           1.0
             & \text{if }f(t) = f(T) \\
           0.0
             & \text{otherwise}
         \end{cases}

      The `not seasonal` version is defined as the constant map.

      .. math::

         q_T(t) = 1.0

    **Feature Map**

      The feature map :math:`f` is configurable. The special case
      :math:`f(t) = [t]` results in the so-called `naive NPTS`. For
      non-seasonal models, by default we have :math:`f(t) = [t]` for the NPTS
      Forecaster (i.e., with the `exponential` kernel) and no features for the
      Climatological Forecaster (i.e., the `uniform` kernel).

      For seasonal NPTS and seasonal Climatological, time features determined
      based on the frequency of the time series are added to the default
      feature map.

      The default time features for various frequencies are

      .. math::

         f(t) =
         \begin{cases}
           [\mathit{MINUTE\_OF\_HOUR}(t)] & \text{for minutely frequency}\\
           [\mathit{HOUR\_OF\_DAY}(t)]    & \text{for hourly frequency}\\
           [\mathit{DAY\_OF\_WEEK}(t)]    & \text{for daily frequency}\\
           [\mathit{DAY\_OF\_MONTH}(t)]   & \text{for weekly frequency}\\
           [\mathit{MONTH\_OF\_YEAR}(t)]  & \text{for monthly frequency}
         \end{cases}

      During prediction, one can provide custom features in `feat_dynamic_real`
      (these have to be defined in both the training and the prediction range).
      If the model is seasonal, these custom features are added to the default
      feature map, otherwise they are ignored. If `feat_dynamic_real` is not
      empty, one can disable default time features by setting the flag
      `use_default_time_features` to `False`.

    Parameters
    ----------

    freq
        time frequency string
    prediction_length
        number of time steps to predict
    context_length
        number of time-steps that are considered before making predictions
        (the default value of None corresponds to the case where all time steps
        in the history are considered)
    kernel_type
        the type of kernel to use (either "exponential" or "uniform")
    exp_kernel_weights
        single weight :math:`\alpha` or the weights for the features to use
        in the exponential kernel; currently, we use the single weight version
        and for seasonal NPTS we just rescale :math:`\alpha` by `feature_scale`
        for seasonal features.
    use_seasonal_model
        whether to use seasonal variant
    use_default_time_features
        time features derived based on the frequency of the time series
    num_default_time_features
        this is not exposed; this parameter is for having more control on the
        number of default time features, as the date_feature_set adds too
        many per default.
    feature_scale
        scale for time (seasonal) features in order to sample past seasons
        with higher probability
    num_eval_samples
        number of prediction samples required
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        kernel_type: KernelType = KernelType.exponential,
        exp_kernel_weights: Union[float, List[float]] = 1.0,
        use_seasonal_model: bool = True,
        use_default_time_features: bool = True,
        num_default_time_features: int = 1,
        num_eval_samples: int = 100,
        feature_scale: float = 1000.0,
    ) -> None:
        self.prediction_length = prediction_length
        self.freq = freq
        self.num_eval_samples = num_eval_samples
        # Similar to lag upper bound in AR2N2 we limit the context length to
        # some maximum value instead of looking at the whole history which
        # might be too large.
        self.context_length = (
            context_length if context_length is not None else 1100
        )
        self.kernel_type = kernel_type
        self.num_default_time_features = num_default_time_features
        self.use_seasonal_model = use_seasonal_model
        self.use_default_time_features = use_default_time_features
        self.feature_scale = feature_scale

        if not self._is_exp_kernel():
            self.kernel = NPTS.uniform_kernel()
        elif isinstance(exp_kernel_weights, float):
            self.kernel = NPTS.log_distance_kernel(exp_kernel_weights)
        elif isinstance(exp_kernel_weights, list):
            self.kernel = NPTS.log_weighted_distance_kernel(exp_kernel_weights)
        else:
            raise RuntimeError(
                'Unexpected "exp_kernel_weights" type - should be either'
                'a float or a list of floats'
            )

    def _is_exp_kernel(self) -> bool:
        return self.kernel_type == KernelType.exponential

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[SampleForecast]:
        for data in dataset:
            start = pd.Timestamp(data['start'])
            target = np.asarray(data['target'], np.float32)
            index = pd.DatetimeIndex(
                start=start, freq=self.freq, periods=len(target)
            )

            # Slice the time series until context_length or history length
            # depending on which ever is minimum
            train_length = min(len(target), self.context_length)
            ts = pd.Series(index=index, data=target)[-train_length:]
            if 'feat_dynamic_real' in data.keys():
                custom_features = np.array(
                    [
                        dynamic_feature[
                            -train_length - self.prediction_length :
                        ]
                        for dynamic_feature in data['feat_dynamic_real']
                    ]
                )
            else:
                custom_features = None

            yield self.predict_time_series(ts, custom_features)

    def predict_time_series(
        self, ts: pd.Series, custom_features: np.ndarray = None
    ) -> SampleForecast:
        """
        Given a training time series, this method generates `Forecast` object
        containing prediction samples for `prediction_length` time points.

        The predictions are generated via weighted sampling where the weights
        are determined by the `NPTSPredictor` kernel type and feature map.

        Parameters
        ----------
        ts
            training time series object
        custom_features
            custom features (covariates) to use

        Returns
        -------
        Forecast
          A prediction for the supplied `ts` and `custom_features`.
        """

        if np.all(np.isnan(ts.values[-self.context_length :])):
            raise GluonTSDataError(
                f'The last {self.context_length} positions of the target time '
                f'series are all NaN. Please increase the `context_length` '
                f'parameter of your NPTS model so the last '
                f'{self.context_length} positions of each target contain at '
                f'least one non-NaN value.'
            )

        # Get the features for both training and prediction ranges
        train_features, predict_features = self._get_features(
            ts.index, self.prediction_length, custom_features
        )

        # Compute weights for sampling for each time step `t` in the
        # prediction range
        sampling_weights_iterator = NPTS.compute_weights(
            train_features=train_features,
            pred_features=predict_features,
            target_isnan_positions=np.argwhere(np.isnan(ts.values)),
            kernel=self.kernel,
            do_exp=self._is_exp_kernel(),
        )

        # Generate forecasts
        forecast = NPTS.predict(
            ts,
            self.prediction_length,
            sampling_weights_iterator,
            self.num_eval_samples,
        )

        return forecast

    def _get_features(
        self,
        train_index: pd.DatetimeIndex,
        prediction_length: int,
        custom_features: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal method for computing default, (optional) seasonal features
        for the training and prediction ranges given time index for the
        training range and the prediction length.

        Appends `custom_features` if provided.

        Parameters
        ----------
        train_index
            Pandas DatetimeIndex
        prediction_length
            prediction length
        custom_features
            shape: (num_custom_features, train_length + pred_length)

        Returns
        -------
        a tuple of (training, prediction) feature tensors
            shape: (num_features, train_length/pred_length)
        """

        train_length = len(train_index)
        full_time_index = pd.date_range(
            train_index.min(),
            periods=train_length + prediction_length,
            freq=train_index.freq,
        )

        # Default feature map for both seasonal and non-seasonal models.
        if self._is_exp_kernel():
            # Default time index features: index of the time point
            # [0, train_length + pred_length - 1]
            features = np.expand_dims(
                np.array(range(len(full_time_index))), axis=0
            )

            # Rescale time index features into the range: [-0.5, 0.5]
            # similar to the seasonal features
            # (see gluonts.time_feature)
            features = features / (train_length + prediction_length - 1) - 0.5
        else:
            # For uniform seasonal model we do not add time index features
            features = np.empty((0, len(full_time_index)))

        # Add more features for seasonal variant
        if self.use_seasonal_model:
            if custom_features is not None:
                total_length = train_length + prediction_length

                assert len(custom_features.shape) == 2, (
                    "Custom features should be 2D-array where the rows "
                    "represent features and columns the time points."
                )

                assert custom_features.shape[1] == total_length, (
                    f"For a seasonal model, feat_dynamic_real must be defined "
                    f"for both training and prediction ranges. They are only "
                    f"provided for {custom_features.shape[1]} time steps "
                    f"instead of {train_length + prediction_length} steps."
                )

                features = np.vstack(
                    [features, self.feature_scale * custom_features]
                )

            if self.use_default_time_features or custom_features is None:
                # construct seasonal features
                seasonal_features_gen = time_features_from_frequency_str(
                    full_time_index.freqstr
                )
                seasonal_features = [
                    self.feature_scale * gen(full_time_index)
                    for gen in seasonal_features_gen[
                        : self.num_default_time_features
                    ]
                ]

                features = np.vstack([features, *seasonal_features])

        train_features = features[:, :train_length]
        pred_features = features[:, train_length:]

        return train_features, pred_features

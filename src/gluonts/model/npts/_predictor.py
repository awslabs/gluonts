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

from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from toolz import take

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.exceptions import GluonTSDataError
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import time_features_from_frequency_str
from gluonts import zebras as zb

from ._model import NPTS


class KernelType(str, Enum):
    exponential = "exponential"
    uniform = "uniform"


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
        feature_scale: float = 1000.0,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        # We limit the context length to some maximum value instead of
        # looking at the whole history which might be too large.
        self.context_length = (
            context_length if context_length is not None else 1100
        )
        self.kernel_type = kernel_type
        self.num_default_time_features = num_default_time_features
        self.use_seasonal_model = use_seasonal_model
        self.use_default_time_features = use_default_time_features
        self.feature_scale = feature_scale
        self.freq = freq

        if not self._is_exp_kernel():
            self.kernel = NPTS.uniform_kernel()
        elif isinstance(exp_kernel_weights, float):
            self.kernel = NPTS.log_distance_kernel(exp_kernel_weights)
        elif isinstance(exp_kernel_weights, list):
            self.kernel = NPTS.log_weighted_distance_kernel(exp_kernel_weights)
        else:
            raise RuntimeError(
                'Unexpected "exp_kernel_weights" type - should be either'
                "a float or a list of floats"
            )

    def _is_exp_kernel(self) -> bool:
        return self.kernel_type == KernelType.exponential

    def predict(
        self, dataset: Dataset, num_samples: int = 100, **kwargs
    ) -> Iterator[SampleForecast]:
        for data in dataset:
            yield self.predict_item(data)

    def predict_item(self, item, num_samples: int = 100):
        target = np.asarray(item["target"], np.float32)

        # Slice the time series until context_length or history length
        # depending on which ever is minimum
        train_length = min(len(target), self.context_length)
        past_target = target[-train_length:]

        index = item["start"].periods(train_length)

        if np.isnan(past_target).all():
            raise GluonTSDataError(
                f"The last {self.context_length} positions of the target time "
                "series are all NaN. Please increase the `context_length` "
                "parameter of your NPTS model so the last "
                f"{self.context_length} positions of each target contain at "
                "least one non-NaN value."
            )

        custom_features: Optional[np.ndarray]
        if "feat_dynamic_real" in item:
            custom_features = item["feat_dynamic_real"][
                ..., -train_length - self.prediction_length :
            ]
        else:
            custom_features = None

        # Get the features for both training and prediction ranges
        train_features, predict_features = self._get_features(
            index, custom_features
        )

        # Compute weights for sampling for each time step `t` in the
        # prediction range
        sampling_weights_iterator = NPTS.compute_weights(
            train_features=train_features,
            pred_features=predict_features,
            target_isnan_positions=np.argwhere(np.isnan(past_target)),
            kernel=self.kernel,
            do_exp=self._is_exp_kernel(),
        )

        return NPTS.predict(
            past_target,
            index,
            prediction_length=self.prediction_length,
            sampling_weights_iterator=sampling_weights_iterator,
            num_samples=num_samples,
            item_id=item.get("item_id"),
        )

    def _get_features(
        self,
        periods: zb.Periods,
        feat_dynamic_real: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        train_length = len(periods)

        full_length = train_length + self.prediction_length
        full_periods = periods.extend(self.prediction_length)

        # Default feature map for both seasonal and non-seasonal models.
        if self._is_exp_kernel():
            # Age feature, scaled from -0.5 to 0.5
            age_feature = np.linspace(-0.5, 0.5, full_length)
        else:
            # For uniform seasonal model we do not add time index features
            age_feature = np.empty(full_length)

        custom_features = [np.empty(full_length)]

        # Add more features for seasonal variant
        if self.use_seasonal_model:
            if feat_dynamic_real is not None:
                assert feat_dynamic_real.ndim == 2
                assert feat_dynamic_real.shape[1] == full_length
                custom_features.append(feat_dynamic_real)

            if self.use_default_time_features:
                custom_features.append(
                    [
                        time_feature(full_periods)
                        for time_feature in take(
                            self.num_default_time_features,
                            time_features_from_frequency_str(periods.freq),
                        )
                    ]
                )

        custom_features = np.vstack(custom_features) * self.feature_scale

        return np.split(
            np.vstack([age_feature, custom_features]),
            [train_length],
            axis=1,
        )

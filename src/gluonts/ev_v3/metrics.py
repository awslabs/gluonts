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

from typing import Optional, Union

import numpy as np

from gluonts.ev_v3.api import Metric
from gluonts.exceptions import GluonTSUserError
from gluonts.model.forecast import Quantile
from gluonts.time_feature import get_seasonality


# This class is here for consistency.
# Other metrics can just use `Target()` instead of `data["target"]`.
class Target(Metric):
    def get_name(self):
        return "target"

    def calculate(self, data):
        return data["target"]


# This class is here for consistency.
# Other metrics can just use `PastData()` instead of `data["past_data"]`.
class PastData(Metric):
    def get_name(self):
        return "past_data"

    def calculate(self, data):
        return data["past_data"]


class Prediction(Metric):
    def __init__(
        self,
        quantile: Optional[Union[Quantile, float, str]] = None,
        use_mean: bool = False,
    ):
        super(Prediction, self).__init__()
        if (quantile is not None) == use_mean:
            raise GluonTSUserError(
                "Either a provided quantile or use_mean=True"
                " was expected, not both"
            )

        self.use_mean = use_mean

        if not use_mean:
            self.quantile = Quantile.parse(quantile)

    def get_name(self):
        if self.use_mean:
            return "Quantile[mean]"
        else:
            return f"Quantile[{self.quantile.name}]"

    def calculate(self, data):
        if self.use_mean:
            return data["forecast_batch"].mean
        else:
            return data["forecast_batch"].quantile(self.quantile.value)


class AbsTarget(Metric):
    def get_name(self):
        return "abs_target"

    def calculate(self, data):
        target = Target().get(data)
        return np.abs(target)


class Error(Metric):
    def __init__(self, error_type: Optional[str] = "median"):
        super(Error, self).__init__()
        self.error_type = self._standardize_error_type(error_type)

    def _standardize_error_type(
        self, error_type: Union[float, str]
    ) -> Union[float, str]:
        # this function returns either a float between 0 and 1
        # to be interpreted as a percentile or "mean"
        if error_type == "mean":
            return "mean"
        if error_type == "median":
            return 0.5
        if error_type.startswith("p"):
            return float(error_type[1:]) / 100
        raise ValueError(
            "error_type must be 'mean', 'median' or a percentile (like 'p90')"
        )

    def get_name(self):
        return f"error[{self.error_type}]"

    def calculate(self, data: dict):
        target = Target().get(data)
        if self.error_type == "mean":
            prediction = Prediction(use_mean=True).get(data)
        else:
            prediction = Prediction(quantile=self.error_type).get(data)

        return target - prediction


class AbsError(Metric):
    def __init__(self, error_type: Optional[str] = "median"):
        super(AbsError, self).__init__()
        self.error_type = error_type

    def get_name(self):
        error_type = self.error_type
        return f"abs_error[{error_type}]"

    def calculate(self, data):
        error = Error(error_type=self.error_type)
        return np.abs(error.get(data))


class SquaredError(Metric):
    def __init__(self, error_type: Optional[str] = "mean"):
        super(SquaredError, self).__init__()
        self.error_type = error_type

    def get_name(self):
        return f"squared_error[{self.error_type}]"

    def calculate(self, data):
        error = Error(error_type=self.error_type)
        return np.square(error.get(data))


class MSE(Metric):
    def __init__(
        self,
        error_type: Optional[str] = "mean",
        axis: Optional[int] = None,
    ):
        super(MSE, self).__init__()
        self.error_type = error_type
        self.axis = axis

    def get_name(self):
        return f"mse[{self.error_type},axis={self.axis}]"

    def calculate(self, data):
        error = Error(self.error_type).get(data)
        return np.mean(error, axis=self.axis)


class RMSE(Metric):
    def __init__(
        self, error_type: Optional[str] = "mean", axis: Optional[int] = None
    ):
        super(RMSE, self).__init__()
        self.error_type = error_type
        self.axis = axis

    def get_name(self):
        return f"rmse[{self.error_type},axis={self.axis}]"

    def calculate(self, data):
        return np.sqrt(
            MSE(error_type=self.error_type, axis=self.axis).get(data)
        )


class NRMSE(Metric):
    def __init__(
        self,
        error_type: Optional[str] = "mean",
        axis: Optional[int] = None,
    ):
        super(NRMSE, self).__init__()
        self.error_type = error_type
        self.axis = axis

    def get_name(self):
        return f"nrmse[{self.error_type},axis={self.axis}]"

    def calculate(self, data):
        return RMSE(self.error_type, self.axis).get(data) / np.mean(
            AbsTarget().get(data), axis=self.axis
        )


class QuantileLoss(Metric):
    def __init__(self, quantile: Union[Quantile, float, str]):
        super(QuantileLoss, self).__init__()
        self.quantile = Quantile.parse(quantile)

    def get_name(self):
        return self.quantile.loss_name

    def calculate(self, data):
        prediction = Prediction(quantile=self.quantile.value).get(data)
        target = Target().get(data)

        return np.abs(
            (target - prediction)
            * ((prediction >= target) - self.quantile.value)
        )


class MAPE(Metric):
    def __init__(self, error_type: Optional[str], axis: Optional[int] = None):
        super(MAPE, self).__init__()
        self.error_type = error_type
        self.axis = axis

    def get_name(self):
        return f"mape[{self.error_type},axis={self.axis}]"

    def calculate(self, data):
        abs_error = AbsError(error_type=self.error_type).get(data)
        abs_target = AbsTarget().get(data)
        return np.mean(abs_error / abs_target, axis=self.axis)


class SMAPE(Metric):
    def __init__(self, error_type: Optional[str], axis: Optional[int] = None):
        super(SMAPE, self).__init__()
        self.error_type = error_type
        self.axis = axis

    def get_name(self):
        return f"smape[{self.error_type},axis={self.axis}]"

    def calculate(self, data):
        abs_error = AbsError(error_type=self.error_type).get(data)
        abs_target = AbsTarget().get(data)
        if self.error_type == "mean":
            prediction = Prediction(use_mean=True).get(data)
        else:
            prediction = Prediction(quantile=self.error_type).get(data)

        return 2 * np.mean(
            abs_error / (abs_target + np.abs(prediction)), axis=self.axis
        )


class ND(Metric):
    def __init__(
        self, error_type: Optional[str] = "median", axis: Optional[int] = None
    ):
        super(ND, self).__init__()
        self.error_type = error_type
        self.axis = axis

    def get_name(self):
        return f"ND[{self.error_type},axis={self.axis}]"

    def calculate(self, data):
        abs_error = AbsError(error_type=self.error_type).get(data)
        abs_target = AbsTarget().get(data)

        return np.sum(abs_error, axis=self.axis) / np.sum(
            abs_target, axis=self.axis
        )


class SeasonalError(Metric):
    def __init__(
        self,
        freq: Optional[str] = None,
        seasonality: Optional[int] = None,
        axis: Optional[int] = None,
    ):
        super(SeasonalError, self).__init__()
        self.axis = axis
        if seasonality:
            self.seasonality = seasonality
        else:
            assert (
                freq is not None
            ), "Either freq or seasonality must be provided"
            self.seasonality = get_seasonality(freq)

    def get_name(self):
        return f"season_error[seasonality={self.seasonality},axis={self.axis}]"

    def calculate(self, data):
        past_data = PastData().get(data)

        if self.seasonality < len(past_data):
            forecast_freq = self.seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            forecast_freq = 1

        # TODO: using a dynamic axis gets ugly here - what can we do?
        if np.ndim(past_data) != 2:
            raise ValueError(
                "Seasonal error can't handle input data"
                " that is not 2-dimensional"
            )
        if self.axis == 0:
            y_t = past_data[:-forecast_freq, :]
            y_tm = past_data[forecast_freq:, :]
        elif self.axis == 1:
            y_t = past_data[:, :-forecast_freq]
            y_tm = past_data[:, forecast_freq:]
        else:
            raise ValueError(
                "Seasonal error can only handle 0 or 1 for axis argument"
            )

        return np.mean(np.abs(y_t - y_tm), axis=self.axis)


class Coverage(Metric):
    pass  # TODO


class MASE(Metric):
    pass  # TODO


class MSIS(Metric):
    pass  # TODO


class WeightedQuantileLoss(Metric):
    pass  # TODO


class OWA(Metric):
    pass  # TODO

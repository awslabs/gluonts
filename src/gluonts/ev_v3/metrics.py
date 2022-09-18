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

from gluonts.ev_v3.api import BaseMetric, AggregateMetric, get_standard_type
from gluonts.model.forecast import Quantile
from gluonts.time_feature import get_seasonality


# --- INPUTS & BASE METRICS (same dimensionality as inputs) ---

# This class allows using `PredictionTarget()` instead of `data["PredictionTarget"]`
class PredictionTarget(BaseMetric):
    def get_name(self):
        return "PredictionTarget"

    def calculate(self, data):
        assert self.name in data, f"Missing input '{self.name}' in data"
        return data[self.name]


class AbsPredictionTarget(BaseMetric):
    def get_name(self):
        return "AbsPredictionTarget"

    def calculate(self, data):
        prediction_target = PredictionTarget().get(data)
        return np.abs(prediction_target)


# This class allows using `InputData()` instead of `data["InputData"]`
class InputData(BaseMetric):
    def get_name(self):
        return "InputData"

    def calculate(self, data):
        assert self.name in data, f"Missing input '{self.name}' in data"
        return data[self.name]


class Prediction(BaseMetric):
    def __init__(
        self,
        prediction_type: Union[Quantile, float, str] = 0.5,
    ):
        super(Prediction, self).__init__()
        self.prediction_type = get_standard_type(prediction_type)

    def get_name(self):
        return f"Prediction[{self.prediction_type}]"

    def calculate(self, data):
        assert "Forecast" in data, f"Missing input 'Forecast' in data"

        if self.prediction_type == "mean":
            return data["Forecast"].mean
        else:
            quantile = Quantile.parse(self.prediction_type)
            return data["Forecast"].quantile(quantile.value)


class AbsPrediction(BaseMetric):
    def __init__(
        self,
        prediction_type: Union[Quantile, float, str] = 0.5,
    ):
        super(AbsPrediction, self).__init__()
        self.prediction_type = get_standard_type(prediction_type)

    def get_name(self):
        return f"AbsPrediction[{self.prediction_type}]"

    def calculate(self, data):
        prediction = Prediction(self.prediction_type).get(data)
        return np.abs(prediction)


class Error(BaseMetric):
    def __init__(
        self,
        error_type: Union[Quantile, float, str] = 0.5,
    ):
        super(Error, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"Error[{self.error_type}]"

    def calculate(self, data: dict):
        prediction_target = PredictionTarget().get(data)
        prediction = Prediction(prediction_type=self.error_type).get(data)

        return prediction_target - prediction


class AbsError(BaseMetric):
    def __init__(
        self,
        error_type: Union[Quantile, float, str] = 0.5,
    ):
        super(AbsError, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"AbsError[{self.error_type}]"

    def calculate(self, data):
        error = Error(error_type=self.error_type).get(data)
        return np.abs(error)


class SquaredError(BaseMetric):
    def __init__(
        self,
        error_type: Union[Quantile, float, str] = "mean",
    ):
        super(SquaredError, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"SquaredError[{self.error_type}]"

    def calculate(self, data):
        error = Error(error_type=self.error_type).get(data)
        return np.square(error)


class QuantileLoss(BaseMetric):
    def __init__(
        self,
        quantile: Union[Quantile, float, str] = 0.5,
    ):
        super(QuantileLoss, self).__init__()
        self.quantile = Quantile.parse(quantile)

    def get_name(self):
        return self.quantile.loss_name

    def calculate(self, data):
        error = Error(error_type=self.quantile).get(data)
        prediction = Prediction(prediction_type=self.quantile).get(data)
        prediction_target = PredictionTarget().get(data)

        return np.abs(
            error * ((prediction >= prediction_target) - self.quantile.value)
        )


class Coverage(BaseMetric):
    def __init__(
        self,
        quantile: Union[Quantile, float, str] = 0.5,
    ):
        super(Coverage, self).__init__()
        self.quantile = Quantile.parse(quantile)

    def get_name(self):
        return self.quantile.coverage_name

    def calculate(self, data):
        prediction_target = PredictionTarget().get(data)
        prediction = Prediction(prediction_type=self.quantile).get(data)

        return prediction_target < prediction


# -- AGGREGATE METRICS (lower dimensionality than input & base metrics) --


class MSE(AggregateMetric):
    def __init__(
        self,
        error_type: Union[Quantile, float, str] = "mean",
    ):
        super(MSE, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"MSE[{self.error_type}]"

    def calculate(self, data, axis):
        squared_error = SquaredError(error_type=self.error_type).get(data)
        return np.mean(squared_error, axis=axis)


class RMSE(AggregateMetric):
    def __init__(self, error_type: Union[Quantile, float, str] = "mean"):
        super(RMSE, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"RMSE[{self.error_type}]"

    def calculate(self, data, axis):
        mse = MSE(error_type=self.error_type).get(data, axis)
        return np.sqrt(mse)


class NRMSE(AggregateMetric):
    def __init__(self, error_type: Union[Quantile, float, str] = "mean"):
        super(NRMSE, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"NRMSE[{self.error_type}]"

    def calculate(self, data, axis):
        rmse = RMSE(error_type=self.error_type).get(data, axis)
        abs_prediction_target = AbsPredictionTarget().get(data)

        return rmse / np.mean(abs_prediction_target, axis=axis)


class MAPE(AggregateMetric):
    def __init__(self, error_type: Union[Quantile, float, str] = 0.5):
        super(MAPE, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"MAPE[{self.error_type}]"

    def calculate(self, data, axis):
        abs_error = AbsError(error_type=self.error_type).get(data)
        abs_prediction_target = AbsPredictionTarget().get(data)

        return np.mean(abs_error / abs_prediction_target, axis)


class SMAPE(AggregateMetric):
    def __init__(self, error_type: Union[Quantile, float, str] = 0.5):
        super(SMAPE, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"sMAPE[{self.error_type}]"

    def calculate(self, data, axis):
        abs_error = AbsError(error_type=self.error_type).get(data)
        abs_prediction_target = AbsPredictionTarget().get(data)
        abs_prediction = AbsPrediction(prediction_type=self.error_type).get(
            data
        )

        return 2 * np.mean(
            abs_error / (abs_prediction_target + abs_prediction),
            axis=axis,
        )


class ND(AggregateMetric):
    def __init__(self, error_type: Union[Quantile, float, str] = 0.5):
        super(ND, self).__init__()
        self.error_type = get_standard_type(error_type)

    def get_name(self):
        return f"ND[{self.error_type}]"

    def calculate(self, data, axis):
        abs_error = AbsError(error_type=self.error_type).get(data)
        abs_prediction_target = AbsPredictionTarget().get(data)

        return np.sum(abs_error, axis=axis) / np.sum(
            abs_prediction_target, axis=axis
        )


class SeasonalError(AggregateMetric):
    def __init__(
        self, freq: Optional[str] = None, seasonality: Optional[int] = None
    ):
        super(SeasonalError, self).__init__()
        if seasonality:
            self.seasonality = seasonality
        else:
            assert (
                freq is not None
            ), "Either freq or seasonality must be provided"
            self.seasonality = get_seasonality(freq)

    def get_name(self):
        return f"SeasonalError[seasonality={self.seasonality}]"

    def calculate(self, data, axis):
        input_data = InputData().get(data)

        if self.seasonality < len(input_data):
            forecast_freq = self.seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            forecast_freq = 1

        # TODO: using a dynamic axis gets ugly here - what can we do?
        if np.ndim(input_data) != 2:
            raise ValueError(
                "Seasonal error can't handle input data"
                " that is not 2-dimensional"
            )
        if axis == 0:
            y_t = input_data[:-forecast_freq, :]
            y_tm = input_data[forecast_freq:, :]
        elif axis == 1:
            y_t = input_data[:, :-forecast_freq]
            y_tm = input_data[:, forecast_freq:]
        else:
            raise ValueError(
                "Seasonal error can only handle 0 or 1 for axis argument"
            )

        return np.mean(np.abs(y_t - y_tm), axis=axis)


class MASE(AggregateMetric):
    def __init__(
        self,
        error_type: Union[Quantile, float, str] = "median",
        freq: Optional[str] = None,
        seasonality: Optional[int] = None,
    ):
        super(MASE, self).__init__()
        self.error_type = get_standard_type(error_type)
        self.freq = freq
        self.seasonality = seasonality

    def get_name(self):
        return (
            f"MASE[{self.error_type},freq={self.freq},"
            f"seasonality={self.seasonality}]"
        )

    def calculate(self, data, axis):
        abs_error = AbsError(error_type=self.error_type).get(data)
        seasonal_error = SeasonalError(
            freq=self.freq, seasonality=self.seasonality
        ).get(data, axis)

        return np.mean(abs_error, axis=axis) / seasonal_error


class MSIS(AggregateMetric):
    def __init__(
        self,
        alpha: float = 0.05,
        freq: Optional[str] = None,
        seasonality: Optional[int] = None,
    ):
        super(MSIS, self).__init__()
        self.alpha = alpha
        self.freq = freq
        self.seasonality = seasonality

    def get_name(self):
        return (
            f"MSIS[alpha={self.alpha}],freq={self.freq},"
            f"seasonality={self.seasonality}]"
        )

    def calculate(self, data, axis) -> np.ndarray:
        lower_quantile = Prediction(prediction_type=self.alpha / 2).get(data)
        upper_quantile = Prediction(prediction_type=1.0 - self.alpha / 2).get(
            data
        )
        prediction_target = PredictionTarget().get(data)
        seasonal_error = SeasonalError(
            freq=self.freq, seasonality=self.seasonality
        ).get(data, axis)

        numerator = np.mean(
            upper_quantile
            - lower_quantile
            + 2.0
            / self.alpha
            * (lower_quantile - prediction_target)
            * (prediction_target < lower_quantile)
            + 2.0
            / self.alpha
            * (prediction_target - upper_quantile)
            * (prediction_target > upper_quantile),
            axis=axis,
        )

        return numerator / seasonal_error


class WeightedQuantileLoss(AggregateMetric):
    pass  # TODO


class OWA(AggregateMetric):
    pass  # TODO

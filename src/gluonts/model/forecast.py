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

import re
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Union, Tuple

import numpy as np
import pandas as pd
import pydantic

from gluonts.core.component import validated
from gluonts.exceptions import GluonTSUserError


class LinearInterpolation:
    """
    Linear interpolation based on datapoints (x_coord, y_coord)

    Parameters
    ----------
    x_coord
        x-coordinates of the data points must be in increasing order.
    y_coord
        y-coordinates of the data points - may be a list of lists.
    tol
        tolerance when performing the division in the linear interpolation.
    """

    def __init__(
        self,
        x_coord: List[float],
        y_coord: List[np.ndarray],
        tol: float = 1e-8,
    ) -> None:
        self.x_coord = x_coord
        assert sorted(self.x_coord) == self.x_coord
        self.y_coord = y_coord
        self.num_points = len(self.x_coord)
        assert (
            self.num_points >= 2
        ), "Need at least two points for linear interpolation."
        self.tol = tol

    def __call__(self, x: float):
        return self.linear_interpolation(x)

    def linear_interpolation(self, x: float) -> np.ndarray:
        """
        If x is out of interpolation range, return smallest or largest value.
        Otherwise, find two nearest points [x_1, y_1], [x_2, y_2] and return
        its linear interpolation.

        y = (x_2 - x)/(x_2 - x_1) * y_1 + (x - x_1)/(x_2 - x_1) * y_2.

        Parameters
        ----------
        x
            x-coordinate to evaluate the interpolated points.

        Returns
        -------
        np.ndarray
            Interpolated values same shape as self.y_coord
        """
        if self.x_coord[0] >= x:
            return self.y_coord[0]
        elif self.x_coord[-1] <= x:
            return self.y_coord[-1]
        else:
            for i, (x1, x2) in enumerate(zip(self.x_coord, self.x_coord[1:])):
                if x1 < x < x2:
                    denominator = x2 - x1 + self.tol
                    return (x2 - x) / denominator * self.y_coord[i] + (
                        x - x1
                    ) / denominator * self.y_coord[i + 1]


class ExponentialTailApproximation:
    """
    Approximate function on tails based on knots and make a inference on query
    point. Can be used for either interpolation or extrapolation on tails.

    Parameters
    ----------
    x_coord
        x-coordinates of the data points must be in increasing order.
    y_coord
        y-coordinates of the data points - may be a higher numpy array.
    tol
        tolerance when performing the division and computing the log in the
        exponential extrapolation.
    """

    def __init__(
        self,
        x_coord: List[float],
        y_coord: List[np.ndarray],
        tol: float = 1e-8,
    ) -> None:
        self.x_coord = x_coord
        assert sorted(self.x_coord) == self.x_coord
        self.y_coord = y_coord
        self.num_points = len(self.x_coord)
        assert (
            self.num_points >= 2
        ), "Need at least two points for exponential approximation."
        self.tol = tol
        (
            self.beta_inv_left,
            self.beta_inv_right,
        ) = self.init_exponential_tail_weights()

    def init_exponential_tail_weights(self) -> Tuple[float, float]:
        """
        Initialize the weight of exponentially decaying tail functions based on
        two extreme points on the left and right, respectively.

        Returns
        -------
        Tuple
            beta coefficient for left and right tails.
        """
        q_log_diff = np.log(
            (self.x_coord[1] + self.tol) / (self.x_coord[0] + self.tol)
            + self.tol
        )
        y_diff_left = self.y_coord[1] - self.y_coord[0]
        beta_inv_left = y_diff_left / q_log_diff

        z_log_diff = np.log(
            (1 - self.x_coord[-2] + self.tol)
            / (1 - self.x_coord[-1] + self.tol)
            + self.tol
        )  # z = 1/(1-q)
        y_diff_right = self.y_coord[-1] - self.y_coord[-2]
        beta_inv_right = y_diff_right / z_log_diff

        return beta_inv_left, beta_inv_right

    def left(self, x: float) -> np.ndarray:
        """
        Return the inference made on exponentially decaying tail functions.

        For left tail, x = exp(beta * (q - alpha))
        For right tail, x = 1 - exp(-beta * (q - alpha))

        E.g. for x = self.x_coord[0] or self.x_coord[1], return value is
        exactly self.y_coord[0] or self.y_coord[1], respectively.

        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.
        """
        return (
            self.beta_inv_left
            * np.log((x + self.tol) / (self.x_coord[1] + self.tol) + self.tol)
        ) + self.y_coord[1]

    def right(self, x: float) -> np.ndarray:
        """
        Return the inference made on exponentially decaying tail functions.

        For left tail, x = exp(beta * (q - alpha))
        For right tail, x = 1 - exp(-beta * (q - alpha))

        E.g. for x = self.x_coord[-1] or self.x_coord[-2] ,
        return value is exactly self.y_coord[-1]
        or self.y_coord[-2] respectively.
        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.
        """
        return (
            self.beta_inv_right
            * np.log(
                (1 - self.x_coord[-2] + self.tol) / (1 - x + self.tol)
                + self.tol
            )
        ) + self.y_coord[-2]

    def tail_range(self, default_left_tail=0.1, default_right_tail=0.9):
        """
        Return an effective range of left and right tails.
        """
        left_tail = max(
            self.x_coord[0],
            min(self.x_coord[1], default_left_tail),
        )
        right_tail = min(
            self.x_coord[-1],
            max(self.x_coord[-2], default_right_tail),
        )
        return left_tail, right_tail


class Quantile(pydantic.BaseModel):
    value: float
    name: str

    @property
    def loss_name(self):
        return f"QuantileLoss[{self.name}]"

    @property
    def weighted_loss_name(self):
        return f"wQuantileLoss[{self.name}]"

    @property
    def coverage_name(self):
        return f"Coverage[{self.name}]"

    @classmethod
    def checked(cls, value: float, name: str) -> "Quantile":
        if not 0 <= value <= 1:
            raise GluonTSUserError(
                f"quantile value should be in [0, 1] but found {value}"
            )

        return Quantile(value=value, name=name)

    @classmethod
    def from_float(cls, quantile: float) -> "Quantile":
        assert isinstance(quantile, float)
        return cls.checked(value=quantile, name=str(quantile))

    @classmethod
    def from_str(cls, quantile: str) -> "Quantile":
        assert isinstance(quantile, str)
        try:
            return cls.checked(value=float(quantile), name=quantile)
        except ValueError:
            m = re.match(r"^p(\d{2})$", quantile)

            if m is None:
                raise GluonTSUserError(
                    'Quantile string should be of the form "p10", "p50", ...'
                    f' or "0.1", "0.5", ... but found {quantile}'
                )
            else:
                quantile_float: float = int(m.group(1)) / 100
                return cls(value=quantile_float, name=str(quantile_float))

    @classmethod
    def parse(cls, quantile: Union["Quantile", float, str]) -> "Quantile":
        """
        Produces equivalent float and string representation of a given quantile
        level.

        >>> Quantile.parse(0.1)
        Quantile(value=0.1, name='0.1')

        >>> Quantile.parse('0.2')
        Quantile(value=0.2, name='0.2')

        >>> Quantile.parse('0.20')
        Quantile(value=0.2, name='0.20')

        >>> Quantile.parse('p99')
        Quantile(value=0.99, name='0.99')

        Parameters
        ----------
        quantile
            Quantile, can be a float a str representing a float e.g. '0.1' or a
            quantile string of the form 'p0.1'.

        Returns
        -------
        Quantile
            A tuple containing both a float and a string representation of the
            input quantile level.
        """
        if isinstance(quantile, Quantile):
            return quantile
        elif isinstance(quantile, float):
            return cls.from_float(quantile)
        else:
            return cls.from_str(quantile)


class Forecast:
    """
    A abstract class representing predictions.
    """

    start_date: pd.Period
    item_id: Optional[str]
    info: Optional[Dict]
    prediction_length: int
    mean: np.ndarray
    _index = None

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        """
        Computes a quantile from the predicted distribution.

        Parameters
        ----------
        q
            Quantile to compute.

        Returns
        -------
        numpy.ndarray
            Value of the quantile across the prediction range.
        """
        raise NotImplementedError()

    def quantile_ts(self, q: Union[float, str]) -> pd.Series:
        return pd.Series(index=self.index, data=self.quantile(q))

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.5)

    @property
    def freq(self):
        return self.start_date.freq

    def plot(
        self,
        prediction_intervals=(50.0, 90.0),
        show_mean=False,
        color="b",
        label=None,
        output_file=None,
        *args,
        **kwargs,
    ):
        """
        Plots the median of the forecast as well as prediction interval bounds
        (requires matplotlib and pandas).

        Parameters
        ----------
        prediction_intervals : float or list of floats in [0, 100]
            Prediction interval size(s). If a list, it will stack the error
            plots for each prediction interval. Only relevant for error styles
            with "ci" in the name.
        show_mean : boolean
            Whether to also show the mean of the forecast.
        color : matplotlib color name or dictionary
            The color used for plotting the forecast.
        label : string
            A label (prefix) that is used for the forecast
        output_file : str or None, default None
            Output path for the plot file. If None, plot is not saved to file.
        args :
            Other arguments are passed to main plot() call
        kwargs :
            Other keyword arguments are passed to main plot() call
        """

        # matplotlib==2.0.* gives errors in Brazil builds and has to be
        # imported locally
        import matplotlib.pyplot as plt

        label_prefix = "" if label is None else label + "-"

        for c in prediction_intervals:
            assert 0.0 <= c <= 100.0

        ps = [50.0] + [
            50.0 + f * c / 2.0
            for c in prediction_intervals
            for f in [-1.0, +1.0]
        ]
        percentiles_sorted = sorted(set(ps))

        def alpha_for_percentile(p):
            return (p / 100.0) ** 0.3

        ps_data = [self.quantile(p / 100.0) for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=self.index.to_timestamp())
        p50_series.plot(color=color, ls="-", label=f"{label_prefix}median")

        if show_mean:
            mean_data = np.mean(self._sorted_samples, axis=0)
            pd.Series(data=mean_data, index=self.index.to_timestamp()).plot(
                color=color,
                ls=":",
                label=f"{label_prefix}mean",
                *args,
                **kwargs,
            )

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            plt.fill_between(
                self.index.to_timestamp(),
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
                *args,
                **kwargs,
            )
            # Hack to create labels for the error intervals. Doesn't actually
            # plot anything, because we only pass a single data point
            pd.Series(
                data=p50_data[:1], index=self.index.to_timestamp()[:1]
            ).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                *args,
                **kwargs,
            )
        if output_file:
            plt.savefig(output_file)

    @property
    def index(self) -> pd.PeriodIndex:
        if self._index is None:
            self._index = pd.period_range(
                self.start_date,
                periods=self.prediction_length,
                freq=self.start_date.freq,
            )
        return self._index

    def dim(self) -> int:
        """
        Returns the dimensionality of the forecast object.
        """
        raise NotImplementedError()

    def copy_dim(self, dim: int):
        """
        Returns a new Forecast object with only the selected sub-dimension.

        Parameters
        ----------
        dim
            The returned forecast object will only represent this dimension.
        """
        raise NotImplementedError()

    def copy_aggregate(self, agg_fun: Callable):
        """
        Returns a new Forecast object with a time series aggregated over the
        dimension axis.

        Parameters
        ----------
        agg_fun
            Aggregation function that defines the aggregation operation
            (typically mean or sum).
        """
        raise NotImplementedError()

    def as_json_dict(self, config: "Config") -> dict:
        result = {}

        if OutputType.mean in config.output_types:
            result["mean"] = self.mean.tolist()

        if OutputType.quantiles in config.output_types:
            quantiles = map(Quantile.parse, config.quantiles)

            result["quantiles"] = {
                quantile.name: self.quantile(quantile.value).tolist()
                for quantile in quantiles
            }

        if OutputType.samples in config.output_types:
            result["samples"] = []

        return result


class SampleForecast(Forecast):
    """
    A `Forecast` object, where the predicted distribution is represented
    internally as samples.

    Parameters
    ----------
    samples
        Array of size (num_samples, prediction_length) (1D case) or
        (num_samples, prediction_length, target_dim) (multivariate case)
    start_date
        start of the forecast
    info
        additional information that the forecaster may provide e.g. estimated
        parameters, number of iterations ran etc.
    """

    @validated()
    def __init__(
        self,
        samples: np.ndarray,
        start_date: pd.Period,
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ) -> None:
        assert isinstance(
            samples, np.ndarray
        ), "samples should be a numpy array"
        assert len(np.shape(samples)) == 2 or len(np.shape(samples)) == 3, (
            "samples should be a 2-dimensional or 3-dimensional array."
            " Dimensions found: {}".format(len(np.shape(samples)))
        )
        self.samples = samples
        self._sorted_samples_value = None
        self._mean = None
        self._dim = None
        self.item_id = item_id
        self.info = info

        assert isinstance(
            start_date, pd.Period
        ), "start_date should be a pandas Period object"
        self.start_date = start_date

    @property
    def _sorted_samples(self):
        if self._sorted_samples_value is None:
            self._sorted_samples_value = np.sort(self.samples, axis=0)
        return self._sorted_samples_value

    @property
    def num_samples(self):
        """
        The number of samples representing the forecast.
        """
        return self.samples.shape[0]

    @property
    def prediction_length(self):
        """
        Time length of the forecast.
        """
        return self.samples.shape[1]

    @property
    def mean(self) -> np.ndarray:
        """
        Forecast mean.
        """
        if self._mean is not None:
            return self._mean
        else:
            return np.mean(self.samples, axis=0)

    @property
    def mean_ts(self) -> pd.Series:
        """
        Forecast mean, as a pandas.Series object.
        """
        return pd.Series(self.mean, index=self.index)

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        q = Quantile.parse(q).value
        sample_idx = int(np.round((self.num_samples - 1) * q))
        return self._sorted_samples[sample_idx, :]

    def copy_dim(self, dim: int) -> "SampleForecast":
        if len(self.samples.shape) == 2:
            samples = self.samples
        else:
            target_dim = self.samples.shape[2]
            assert dim < target_dim, (
                f"must set 0 <= dim < target_dim, but got dim={dim},"
                f" target_dim={target_dim}"
            )
            samples = self.samples[:, :, dim]

        return SampleForecast(
            samples=samples,
            start_date=self.start_date,
            item_id=self.item_id,
            info=self.info,
        )

    def copy_aggregate(self, agg_fun: Callable) -> "SampleForecast":
        if len(self.samples.shape) == 2:
            samples = self.samples
        else:
            # Aggregate over target dimension axis
            samples = agg_fun(self.samples, axis=2)
        return SampleForecast(
            samples=samples,
            start_date=self.start_date,
            item_id=self.item_id,
            info=self.info,
        )

    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        else:
            if len(self.samples.shape) == 2:
                # univariate target
                # shape: (num_samples, prediction_length)
                return 1
            else:
                # multivariate target
                # shape: (num_samples, prediction_length, target_dim)
                return self.samples.shape[2]

    def as_json_dict(self, config: "Config") -> dict:
        result = super().as_json_dict(config)

        if OutputType.samples in config.output_types:
            result["samples"] = self.samples.tolist()

        return result

    def __repr__(self):
        return ", ".join(
            [
                f"SampleForecast({self.samples!r})",
                f"{self.start_date!r}",
                f"item_id={self.item_id!r}",
                f"info={self.info!r})",
            ]
        )

    def to_quantile_forecast(self, quantiles: List[str]) -> "QuantileForecast":
        return QuantileForecast(
            forecast_arrays=np.array(
                [
                    self.quantile(q) if q != "mean" else self.mean()
                    for q in quantiles
                ]
            ),
            start_date=self.start_date,
            forecast_keys=quantiles,
            item_id=self.item_id,
            info=self.info,
        )


class QuantileForecast(Forecast):
    """
    A Forecast that contains arrays (i.e. time series) for quantiles and mean.

    Parameters
    ----------
    forecast_arrays
        An array of forecasts
    start_date
        start of the forecast
    forecast_keys
        A list of quantiles of the form '0.1', '0.9', etc.,
        and potentially 'mean'. Each entry corresponds to one array in
        forecast_arrays.
    info
        additional information that the forecaster may provide e.g. estimated
        parameters, number of iterations ran etc.
    """

    def __init__(
        self,
        forecast_arrays: np.ndarray,
        start_date: pd.Period,
        forecast_keys: List[str],
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ) -> None:
        self.forecast_array = forecast_arrays
        assert isinstance(
            start_date, pd.Period
        ), "start_date should be a pandas Period object"
        self.start_date = start_date

        # normalize keys
        self.forecast_keys = [
            Quantile.from_str(key).name if key != "mean" else key
            for key in forecast_keys
        ]
        self.item_id = item_id
        self.info = info
        self._dim = None

        shape = self.forecast_array.shape
        assert shape[0] == len(self.forecast_keys), (
            f"The forecast_array (shape={shape} should have the same "
            f"length as the forecast_keys (len={len(self.forecast_keys)})."
        )
        self.prediction_length = shape[-1]
        self._forecast_dict = {
            k: self.forecast_array[i] for i, k in enumerate(self.forecast_keys)
        }
        self._nan_out = np.array([np.nan] * self.prediction_length)

    def quantile(self, inference_quantile: Union[float, str]) -> np.ndarray:
        sorted_forecast_dict = dict(sorted(self._forecast_dict.items()))
        sorted_forecast_dict.pop("mean", None)
        quantiles = [float(q) for q in sorted_forecast_dict.keys()]
        quantile_predictions = list(sorted_forecast_dict.values())

        inference_quantile = Quantile.parse(inference_quantile).value

        if len(quantiles) == 1 or inference_quantile in quantiles:
            q_str = Quantile.parse(inference_quantile).name
            return self._forecast_dict.get(q_str, self._nan_out)

        linear_interpolation = LinearInterpolation(
            quantiles, quantile_predictions
        )
        exp_tail_approximation = ExponentialTailApproximation(
            quantiles, quantile_predictions
        )
        # The effective range of left, right tails varies over tail
        # approximation class
        (
            left_tail_quantile,
            right_tail_quantile,
        ) = exp_tail_approximation.tail_range()

        if inference_quantile <= left_tail_quantile:
            return exp_tail_approximation.left(inference_quantile)
        elif inference_quantile >= right_tail_quantile:
            return exp_tail_approximation.right(inference_quantile)
        else:
            return linear_interpolation(inference_quantile)

    @property
    def mean(self) -> np.ndarray:
        """
        Forecast mean.
        """
        if "mean" in self._forecast_dict:
            return self._forecast_dict["mean"]

        return self.quantile("p50")

    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        else:
            if (
                len(self.forecast_array.shape) == 2
            ):  # 1D target. shape: (num_samples, prediction_length)
                return 1
            else:
                # 2D target. shape: (num_samples, target_dim,
                # prediction_length)
                return self.forecast_array.shape[1]

    def __repr__(self):
        return ", ".join(
            [
                f"QuantileForecast({self.forecast_array!r})",
                f"start_date={self.start_date!r}",
                f"forecast_keys={self.forecast_keys!r}",
                f"item_id={self.item_id!r}",
                f"info={self.info!r})",
            ]
        )

    def plot(self, label=None, output_file=None, keys=None, *args, **kwargs):
        import matplotlib.pyplot as plt

        label_prefix = "" if label is None else label + "-"

        if keys is None:
            keys = self.forecast_keys

        for k, v in zip(keys, self.forecast_array):
            pd.Series(data=v, index=self.index.to_timestamp()).plot(
                label=f"{label_prefix}q{k}",
                *args,
                **kwargs,
            )
        if output_file:
            plt.savefig(output_file)


class OutputType(str, Enum):
    mean = "mean"
    samples = "samples"
    quantiles = "quantiles"


class Config(pydantic.BaseModel):
    num_samples: int = pydantic.Field(100, alias="num_eval_samples")
    output_types: Set[OutputType] = {OutputType.quantiles, OutputType.mean}
    # FIXME: validate list elements
    quantiles: List[str] = ["0.1", "0.5", "0.9"]

    class Config:
        allow_population_by_field_name = True
        # store additional fields
        extra = "allow"

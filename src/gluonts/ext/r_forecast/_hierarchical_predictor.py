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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.model.forecast import SampleForecast

from . import RBasePredictor


HIERARCHICAL_POINT_FORECAST_METHODS = [
    "naive_bottom_up",
    "top_down_w_average_historical_proportions",
    "top_down_w_proportions_of_the_historical_averages",
    "top_down_w_forecasts_proportions",
    "middle_out_w_forecasts_proportions",
    "mint",
    "erm",
]

HIERARCHICAL_SAMPLE_FORECAST_METHODS = []  # TODO: Add `depbu_mint`.

SUPPORTED_HIERARCHICAL_METHODS = (
    HIERARCHICAL_POINT_FORECAST_METHODS + HIERARCHICAL_SAMPLE_FORECAST_METHODS
)


class RHierarchicalForecastPredictor(RBasePredictor):
    """
    Wrapper for calling the `R hts package
    <https://www.r-pkg.org/pkg/hts>`_.

    In order to use it you need to install R and rpy2. You also need the R `hts` package which
    can be installed by running:

        R -e 'install.packages(c("hts"), repos="https://cloud.r-project.org")'

    Parameters
    ----------
    freq
        The granularity of the time series (e.g. '1H')
    prediction_length
        Number of time points to be predicted.
    is_hts
        Is the time series a hierarchical one as opposed to a grouped time series. # noqa
    target_dim
        The dimension (size) of the multivariate target time series.
    num_bottom_ts
        Number of bottom time series in the hierarchy.
    nodes
        Node structure representing the hierarchichy as defined in the hts package.
        To know the exact strucutre of nodes see the help:
        Hierarhical: https://stackoverflow.com/questions/13579292/how-to-use-hts-with-multi-level-hierarchies
        Grouped: https://robjhyndman.com/hyndsight/gts/
    nonnegative
        Is the target non-negative?
    method_name
        Hierarchical forecasting or reconciliation method to be used; mutst be one of:
        "naive_bottom_up", "middle_out_w_forecasts_proportions", "top_down_w_average_historical_proportions",
        "top_down_w_proportions_of_the_historical_averages", "top_down_w_forecasts_proportions", "mint", "erm"
    fmethod
        The forecasting method to be used for generating base forecasts (i.e., un-reconciled forecasts).
    period
        The period to be used (this is called `frequency` in the R forecast
        package), result to a tentative reasonable default if not specified
        (for instance 24 for hourly freq '1H')
    trunc_length
        Maximum history length to feed to the model (some models become slow
        with very long series).
    params
        Parameters to be used when calling the forecast method default.
        Note that, as `output_type`, only 'samples' is supported currently.
    level
        Level of hierarchy to be used as reference for `middle out` reconciliation (i.e. level=1 means that the level
        below the highest one will be used as reference to compute the forecasts of all the other levels). This value
        is required only for `middle out`.
    numcores
        Number of cores to be used for parallelization of ERM. If not provided, all cores will be used.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        is_hts: bool,
        target_dim: int,
        num_bottom_ts: int,
        nodes: List,
        method_name: str,
        fmethod: str,
        period: Optional[int] = None,
        trunc_length: Optional[int] = None,
        save_info: bool = False,
        nonnegative: bool = False,
        level: Optional[int] = None,
        algorithm: Optional[str] = "cg",
        covariance: Optional[str] = "shr",
        numcores: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            period=period,
            trunc_length=trunc_length,
            save_info=save_info,
            r_file_prefix="hierarchical",
        )

        assert method_name in SUPPORTED_HIERARCHICAL_METHODS, (
            f"method {method_name} is not supported please "
            f"use one of {SUPPORTED_HIERARCHICAL_METHODS}"
        )

        self.method_name = method_name

        self._hts_pkg = self._rpackages.importr("hts")
        self._r_method = self._robjects.r[method_name]

        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_bottom_ts = num_bottom_ts
        self.nodes = nodes
        self.params = {
            "prediction_length": self.prediction_length,
            "output_types": ["samples"],
            "num_samples": 100,
            "frequency": self.period,
            "fmethod": fmethod,
            "nonnegative": nonnegative,
            "algorithm": algorithm,
            "covariance": covariance,
            "numcores": numcores,
            "level": level,
        }

        for p in ["algorithm", "covariance", "numcores", "level"]:
            if self.params[p] is None:
                self.params[p] = self._robjects.rinterface.NULL

        if params is not None:
            self.params.update(params)

        self.is_hts = is_hts

    def _get_r_forecast(self, data: Dict) -> Dict:
        r_params = self._robjects.vectors.ListVector(self.params)

        # R methods take only bottom level time series and the
        # hierarchical (or grouping) structure in the form of
        # `nodes` (or `groups`).
        # First create these `nodes` (or `groups`) as R objects.
        if self.is_hts:
            nodes = self._robjects.ListVector.from_length(len(self.nodes))
            for idx, elem in enumerate(self.nodes):
                if not isinstance(elem, list):
                    elem = [elem]
                nodes[idx] = self._robjects.IntVector(elem)
        else:
            nodes_temp = []
            for idx, elem in enumerate(self.nodes):
                nodes_temp.extend(elem)
            nodes = self._robjects.r.matrix(
                self._robjects.IntVector(nodes_temp),
                ncol=len(self.nodes),
                nrow=len(self.nodes[0]),
                byrow=False,
            ).transpose()

        # Create the bottom level time series as an R object.
        num_ts, nobs = data["target"].shape

        y_bottom = self._robjects.r.matrix(
            self._robjects.FloatVector(data["target"].flatten()),
            ncol=nobs,
            nrow=num_ts,
            byrow=True,
        ).transpose()

        y_bottom_ts = self._stats_pkg.ts(y_bottom, frequency=self.period)

        # Create the hierarchical/grouped time series as an R object.
        if self.is_hts:
            hier_ts = self._hts_pkg.hts(y_bottom_ts, nodes=nodes)
        else:
            hier_ts = self._hts_pkg.gts(y_bottom_ts, groups=nodes)

        forecast = self._r_method(hier_ts, r_params)

        all_forecasts = list(forecast)
        if self.method_name in HIERARCHICAL_POINT_FORECAST_METHODS:
            assert (
                len(all_forecasts) == self.target_dim * self.prediction_length
            )
            hier_point_forecasts = np.reshape(
                list(forecast), (self.target_dim, self.prediction_length)
            ).transpose()
            hier_forecasts = np.tile(
                hier_point_forecasts, (self.params["num_samples"], 1, 1)
            )
        else:
            hier_forecasts = np.reshape(
                list(forecast),
                (
                    self.params["num_samples"],
                    self.prediction_length,
                    self.target_dim,
                ),
                order="F",
            )
        forecast_dict = dict(samples=hier_forecasts)

        return forecast_dict

    def _preprocess_data(self, data: Dict) -> Dict:
        # R methods accept only the bottom level time series and construct
        # aggregated time series using the provided aggregation structure.
        data["target"] = data["target"][-self.num_bottom_ts :, :]

        if self.trunc_length:
            shift_by = max(data["target"].shape[1] - self.trunc_length, 0)
            data["start"] = data["start"] + shift_by
            data["target"] = data["target"][:, -self.trunc_length :]

        return data

    def _forecast_dict_to_obj(
        self,
        forecast_dict: Dict,
        forecast_start_date: pd.Timestamp,
        item_id: Optional[str],
        info: Dict,
    ) -> SampleForecast:
        samples = np.array(forecast_dict["samples"])

        expected_shape = (
            self.params["num_samples"],
            self.prediction_length,
            self.target_dim,
        )
        assert (
            samples.shape == expected_shape
        ), f"Expected shape {expected_shape} but found {samples.shape}"

        return SampleForecast(
            samples,
            start_date=forecast_start_date,
            info=info,
            item_id=item_id,
        )

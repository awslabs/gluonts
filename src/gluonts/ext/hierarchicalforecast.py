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

from typing import List, Optional, Union, Any, Dict

import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.core import _build_fn_name

from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import QuantileForecast
from gluonts.ext.statsforecast import ModelConfig

models_without_fitted_capability = [
    "ADIDA",
    "CrostonClassic",
    "CrostonOptimized",
    "CrostonSBA",
    "IMAPA",
    "SeasWA",
    "TSB",
    "WindowAverage",
]


def get_formatted_S(
    _S: Union[List[List[int]], np.ndarray],
    ts_names: List[str],
) -> pd.DataFrame:
    """
    We format the summation matrix S as a dataframe,
    where the index and columns have the
    corresponding time series names.
    """

    S = np.array(_S)

    return pd.DataFrame(
        S, index=ts_names, columns=ts_names[S.shape[0] - S.shape[1] :]
    )


def format_data_entry(entry: DataEntry, S: pd.DataFrame) -> pd.DataFrame:
    """
    Format data entry as required by hierarchicalforecast.

    ``entry`` is a dictionary with keys: ``"start"``, ``"item_id"``, ``"target"``.
    ``entry["target"]`` is a ``np.ndarray`` with shape ``(num_ts, num_timestamps)``,
    and each row corresponds to one time series of the hierarchy.
    The goal is to reshape this DataEntry as a dataframe where:
    1) the index corresponds to the name of the time series,
    2) the columns ``"ds"`` and ``"y"`` correspond to timestamps and actuals, respectively.
    """

    df = pd.DataFrame(entry["target"]).T
    df.columns = S.index.tolist()
    df.index = pd.date_range(
        start=entry["start"].start_time,
        periods=df.shape[0],
        freq=entry["start"].freq,
    )

    df = unpivot(df)
    df = df.set_index("unique_id")

    return df


def unpivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpivot data frame.

    The input dataframe has as index the time stamps,
    and one column per each time series of the hierarchy.
    We unpivot this so that the final dataframe has
    three columns, i.e. ``"unique_id"``, ``"ds"``, and ``"y"``, where
    1) ``"unique_id"`` has the name of the corresponding time series,
    2) ``"ds"`` has the corresponding time stamps,
    3) ``"y"`` has the actuals.
    """

    n, k = df.shape
    return pd.DataFrame(
        {
            "unique_id": np.asarray(df.columns).repeat(n),
            "ds": np.tile(np.asarray(df.index), k),
            "y": df.to_numpy().ravel("F"),
        }
    )


def format_reconciled_forecasts(
    df: pd.DataFrame,
    prediction_length: int,
    fcst_col_name: str,
    S: pd.DataFrame,
):
    target_dim = S.shape[0]

    assert len(df) == target_dim * prediction_length

    # we want rows to correspond to time and columns to forecasts
    hier_forecasts = df.pivot(columns="ds", values=fcst_col_name)

    # we want columns to follow the order of the rows of S
    hier_forecasts = hier_forecasts.loc[S.index].T

    # we want rows to be sorted according to time
    hier_forecasts = hier_forecasts.sort_index()

    return np.array(hier_forecasts)


def prune_fcst_df(
    df: pd.DataFrame, base_reconciliation_model_name: str
) -> pd.DataFrame:
    # keep certain columns
    columns_to_keep = ["ds"]
    columns_to_keep.extend(
        [x for x in df.columns if base_reconciliation_model_name in x]
    )
    df = df[columns_to_keep]

    # rename columns
    mapper = {
        e: (
            "mean"
            if e == base_reconciliation_model_name
            else e.replace(f"{base_reconciliation_model_name}-", "")
        )
        for e in df.columns
    }
    df = df.rename(columns=mapper)

    return df


class HierarchicalForecastPredictor(RepresentablePredictor):
    """
    A predictor that wraps models from the `hierarchicalforecast`_ package.

    .. hierarchicalforecast: https://github.com/Nixtla/hierarchicalforecast

    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    base_model
        forecaster to use for base forecasts. Please refer to the documentation
        of ``statsforecast`` for available options.
        Example: AutoARIMA
    reconciler
        forecast reconciliation method to use. Please see the documentation
        of ``hierarchicalforecast`` for available options.
        Example: BottomUp
    S
        Summation or aggregation matrix.
    tags
        Each key is a level with values of tags associated to that level
    ts_names
        ordered list with names of time series
    intervals_method
        method used to calculate prediction intervals.
        Options are `normality`, `bootstrap`, `permbu`.
    quantile_levels
        Optional list of quantile levels that we want predictions for.
        Note: this is only supported by specific types of models, such as
        ``AutoARIMA``. By default this is ``None``, giving only the mean
        prediction.
    n_jobs
        number of jobs used in the parallel processing, use -1 for all cores.
    model_params
        dictionary with inputs parameters for base_model,
        i.e. {"season_length": 2}
    reconciler_params
        dictionary with input parameters for reconciler,
        i.e. {"method": "average_proportions"}
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        base_model: Any,
        reconciler: Any,
        S: Union[List[List[int]], np.ndarray],
        tags: Dict[str, Union[List, np.ndarray]] = {},
        ts_names: Optional[List[str]] = None,
        intervals_method: str = "normality",
        quantile_levels: Optional[List[float]] = None,
        n_jobs: int = 1,
        model_params: Dict[str, Any] = {},
        reconciler_params: Dict[str, Any] = {},
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        assert intervals_method in ["normality", "bootstrap", "permbu"]

        if tags and ts_names:
            assert set([x for values in tags.values() for x in values]) == set(
                ts_names
            ), "tags and ts_names must have the same set of ts names"

        self.models = [base_model(**model_params)]
        self.hrec = HierarchicalReconciliation(
            reconcilers=[reconciler(**reconciler_params)]
        )
        self.ts_names = (
            ts_names
            if ts_names is not None
            else [str(x) for x in range(len(S))]
        )
        self.S = get_formatted_S(S, self.ts_names)
        self.tags = {key: np.array(val) for key, val in tags.items()}
        self.intervals_method = intervals_method
        self.n_jobs = n_jobs
        self.config = ModelConfig(
            quantile_levels=quantile_levels,
        )
        self.base_reconciliation_model_name = (
            f"{repr(self.models[0])}/"
            f"{_build_fn_name(self.hrec.reconcilers[0])}"
        )
        self.fitted = (
            repr(self.models[0]) not in models_without_fitted_capability
        )

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        kwargs = {}
        if self.config.intervals is not None and all(
            [
                proportion not in _build_fn_name(self.hrec.reconcilers[0])
                for proportion in [
                    "forecast_proportions",
                    "average_proportions",
                    "proportion_averages",
                ]
            ]
        ):
            kwargs["level"] = self.config.intervals

        Y_df = format_data_entry(entry, self.S)

        # set up forecaster
        forecaster = StatsForecast(
            df=Y_df,
            models=self.models,
            freq=entry["start"].freq,
            n_jobs=self.n_jobs,
        )

        # compute base forecasts
        Y_hat_df = forecaster.forecast(
            h=self.prediction_length, fitted=self.fitted, **kwargs
        )

        params = dict(
            Y_hat_df=Y_hat_df,
            S=self.S,
            tags=self.tags,
            intervals_method=self.intervals_method,
            **kwargs,
        )

        if self.fitted:
            params["Y_df"] = forecaster.forecast_fitted_values()
        else:
            params["Y_df"] = Y_df

        # reconcile forecasts
        Y_hat_df_rec = self.hrec.reconcile(**params)

        # select relevant columns and rename
        Y_hat_df_rec = prune_fcst_df(
            Y_hat_df_rec, self.base_reconciliation_model_name
        )

        # if only mean fcst is computed, we take it as all requested quantiles
        if len(Y_hat_df_rec.columns) == 2 and all(
            Y_hat_df_rec.columns == ["ds", "mean"]
        ):
            fcst_col_names = ["mean"] * len(self.config.statsforecast_keys)
        else:
            fcst_col_names = self.config.statsforecast_keys

        # prepare for QuantileForecast format
        forecast_arrays = np.array(
            [
                format_reconciled_forecasts(
                    df=Y_hat_df_rec,
                    fcst_col_name=fcst_col_names[e],
                    prediction_length=self.prediction_length,
                    S=self.S,
                )
                for e, k in enumerate(self.config.statsforecast_keys)
            ]
        )

        return QuantileForecast(
            forecast_arrays=forecast_arrays,
            forecast_keys=self.config.forecast_keys,
            start_date=forecast_start(entry),
            item_id=entry.get("item_id"),
            info=entry.get("info"),
        )

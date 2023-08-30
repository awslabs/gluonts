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

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tsbench.analysis.utils import (
    loocv_split,
    num_fitting_processes,
    run_parallel,
)
from tsbench.config import Config, ModelConfig
from tsbench.evaluations.metrics import Performance
from tsbench.evaluations.tracking import ModelTracker
from tsbench.surrogate import AutoGluonSurrogate, Surrogate
from .metrics import mrr, ndcg, nrmse, precision_k, smape


class SurrogateAnalyzer:
    """
    The surrogate analyzer evaluates the performance of a surrogate model with
    respect to ranking and regression metrics.

    The analysis is run in parallel and should, thus, not be used in a Jupyter
    notebook. Instead, consider using the `tsbench` CLI.
    """

    def __init__(
        self,
        surrogate: Surrogate[ModelConfig],
        tracker: ModelTracker,
        metrics: Optional[List[str]] = None,
    ):
        """
        Args:
            surrogate: The surrogate model to evaluate.
            tracker: The collector from which to obtain the data for evaluation.
            metrics: The metrics to evaluate. If not provided, evaluates all metrics.
        """
        self.surrogate = surrogate
        self.tracker = tracker
        self.metrics = metrics

    def run(self) -> pd.DataFrame:
        """
        Runs the evaluation on the surrogate by applying LOOCV on the datasets
        being trained on. Metrics are then provided per test dataset.

        Returns:
            A data frame with the results for each fold, the metrics being the columns. The rows
                are indexed by the dataset which was left out.
        """
        if isinstance(self.surrogate, AutoGluonSurrogate):
            metrics = [
                self._run_on_dataset(x)
                for x in tqdm(list(loocv_split(self.tracker)))
            ]
        else:
            data = list(loocv_split(self.tracker))
            metrics = run_parallel(
                self._run_on_dataset,
                data=data,
                num_processes=min(
                    num_fitting_processes(
                        cpus_per_process=self.surrogate.required_cpus,
                        memory_per_process=self.surrogate.required_memory,
                    ),
                    len(data),
                ),
            )
        return pd.concat(metrics).set_index("test_dataset")

    def _run_on_dataset(
        self,
        data: Tuple[
            Tuple[List[Config[ModelConfig]], List[Performance]],
            Tuple[List[Config[ModelConfig]], List[Performance]],
        ],
    ) -> pd.DataFrame:
        (X_train, y_train), (X_test, y_test) = data

        # Fit model and predict
        self.surrogate.fit(X_train, y_train)
        y_pred = self.surrogate.predict(X_test)

        # Compute metrics
        scores = self._score(y_pred, y_test)
        return scores.assign(test_dataset=X_test[0].dataset.name())

    def _score(
        self, y_pred: List[Performance], y_true: List[Performance]
    ) -> pd.DataFrame:
        df_pred = Performance.to_dataframe(y_pred)
        df_true = Performance.to_dataframe(y_true)

        if self.metrics is not None:
            df_pred = df_pred[self.metrics]
            df_true = df_true[self.metrics]

        # We extract the NumPy arrays so that indexing is easier. Each metric is computed such that
        # it results in an array of shape [D] where D is the number of metrics.
        columns = df_pred.columns
        y_pred_min = df_pred.to_numpy()  # type: ignore
        y_true_min = df_true.to_numpy()  # type: ignore

        # Return all results
        metrics = {
            "nrmse": nrmse(y_pred_min, y_true_min),
            "smape": smape(y_pred_min, y_true_min),
            "mrr": mrr(y_pred_min, y_true_min),
            **{
                f"precision_{k}": precision_k(k, y_pred_min, y_true_min)
                for k in (5, 10, 20)
            },
            "ndcg": ndcg(y_pred_min, y_true_min),
        }
        column_index = pd.MultiIndex.from_tuples(
            [(c, m) for m in sorted(metrics) for c in columns]
        )
        values = np.concatenate([metrics[m] for m in sorted(metrics)])
        return pd.DataFrame(np.reshape(values, (1, -1)), columns=column_index)

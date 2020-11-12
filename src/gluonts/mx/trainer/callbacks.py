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

# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np
import mxnet.gluon.nn as nn

# First-party imports
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from gluonts.support.util import copy_parameters


# later write abstract base class and make cbs a list so that a list of callbacks is executed make methods pass
# methods should be post initiliaze post epoch like keras
# MAYBE NOT PUT INTO MX.
class Callback:
    pass


class EarlyStopping(Callback):
    def __init__(
        self,
        validation_dataset,
        predictor: Predictor,
        evaluator: Evaluator = Evaluator(),
        metric: str = "MSE",
        patience: int = 10,
        min_delta: Optional[float] = 0.0,
        verbose: bool = True,
        minimize_metric: bool = True,
        restore_best_network: bool = True,
        num_samples: int = 100,
    ):
        assert (
            patience >= 0
        ), "EarlyStopping Callback patience needs to be >= 0"
        assert (
            min_delta >= 0
        ), "EarlyStopping Callback min_delta needs to be >= 0.0"
        assert (
            num_samples >= 1
        ), "EarlyStopping Callback num_samples needs to be >= 1"

        self.validation_dataset = validation_dataset
        self.predictor = predictor
        self.evaluator = evaluator
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_network = restore_best_network
        self.num_samples = num_samples

        if minimize_metric:
            self.best_metric_value = np.inf
            self.is_better = np.less
        else:
            self.best_metric_value = -np.inf
            self.is_better = np.greater

        self.validation_metric_history = []
        self.best_network = None
        self.n_stale_epochs = 0

    def __call__(self, epoch_no: int, training_network: nn.HybridBlock, **kwargs):
        should_continue = True
        copy_parameters(training_network, self.predictor.prediction_net)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.validation_dataset,
            predictor=self.predictor,
            num_samples=self.num_samples,
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)

        agg_metrics, item_metrics = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(self.validation_dataset)
        )
        current_metric_value = agg_metrics[self.metric]
        self.validation_metric_history.append(current_metric_value)

        if self.verbose:
            print(
                f"Validation metric {self.metric}: {current_metric_value}, best: {self.best_metric_value}"
            )

        if self.is_better(current_metric_value, self.best_metric_value):
            self.best_metric_value = current_metric_value

            if self.restore_best_network:
                training_network.save_parameters("best_network.params")

            self.n_stale_epochs = 0
        else:
            self.n_stale_epochs += 1
            if self.n_stale_epochs == self.patience:
                should_continue = False
                print(
                    f"EarlyStopping callback initiated stop of training at epoch {epoch_no}."
                )

                if self.restore_best_network:
                    print(
                        f"Restoring best network from epoch {epoch_no-self.patience}."
                    )
                    training_network.load_parameters("best_network.params")

        return should_continue


class LoggingCallback(Callback):
    pass


class TerminateOnNaN(Callback):
    pass


class WarmStart(Callback):
    pass

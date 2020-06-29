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
from typing import Any, Dict, Optional

import mxnet.gluon.nn as nn

# First-party imports
from gluonts.core.component import validated


class IterationAveragingStrategy:
    @validated()
    def __init__(self):
        r"""
        Parameters
        ----------
        averaged_model
            Dict that maintains the averaged model parameters.
        cached_model
            Temporarily save the current model, so that the averaged model can be used for validation.
        average_counter
            The number of models accumulated in the average.
        averaging_started
            Indicate whether the model averaging has started.
        """

        self.averaged_model = None
        self.cached_model = None
        self.average_counter = 0
        self.averaging_started = False

    def update_average_trigger(self, average_trigger: Any):
        r"""
        Parameters
        ----------
        average_trigger
            The criteria to trigger averaging.

        Returns
        -------
        """
        # implement a naive strategy, use average_trigger as boolean
        self.averaging_started = average_trigger
        # raise NotImplementedError()

    def apply(self, model: nn.HybridBlock) -> Optional[Dict]:
        r"""
        Parameters
        ----------
        model
            The model of the current iteration.

        Returns
        -------
        The averaged model, None if the averaging hasn't started.
        """

        if self.averaging_started:
            self.update_average(model)

        return self.averaged_model

    def update_average(self, model: nn.HybridBlock):
        r"""
        Parameters
        ----------
        model
            The model to update the average.
        """
        self.average_counter += 1
        if self.averaged_model is None:
            self.averaged_model = {
                k: v.list_data()[0].copy()
                for k, v in model.collect_params().items()
            }
        else:
            alpha = 1.0 / self.average_counter
            # moving average
            for name, param_avg in self.averaged_model.items():
                param_avg[:] += alpha * (
                    model.collect_params()[name].list_data()[0] - param_avg
                )

    def load_averaged_model(self, model: nn.HybridBlock):
        r"""
        When validating/evaluating the averaged model in the half way of training, 
        use load_averaged_model first to load the averaged model and overwrite the current model, 
        do the evaluation, and then use load_cached_model to load the current model back.

        Parameters
        ----------
        model
            The model that the averaged model is loaded to.
        """
        if self.averaged_model is not None:
            # cache the current model
            if self.cached_model is None:
                self.cached_model = {
                    k: v.list_data()[0].copy()
                    for k, v in model.collect_params().items()
                }
            else:
                for name, param_cached in self.cached_model.items():
                    param_cached[:] = model.collect_params()[name].list_data()[
                        0
                    ]
            # load the averaged model
            for name, param_avg in self.averaged_model.items():
                model.collect_params()[name].set_data(param_avg)

    def load_cached_model(self, model: nn.HybridBlock):
        r"""
        Parameters
        ----------
        model
            The model that the cached model is loaded to.
        """
        if self.cached_model is not None:
            # load the cached model
            for name, param_cached in self.cached_model.items():
                model.collect_params()[name].set_data(param_cached)


class NTA_V1(IterationAveragingStrategy):
    @validated()
    def __init__(self, n: int = 5, maximize: bool = False):
        r"""
        Depending on the choice of metrics, the users may want to minimize or maximize the metrics.
        Thus, set maximize = True to maximize, otherwise minimize.

        Parameters
        ----------
        n
            The non-montone interval.
        maximize
            Whether to maximize or minimize the validation metric.
        val_logs
            Historical validation metrics.
        """

        super().__init__()

        self.n = n
        self.maximize = maximize
        self.val_logs = []

    def update_average_trigger(self, average_trigger: Any):
        r"""
        Parameters
        ----------
        average_trigger
            The criteria to trigger averaging, evaluation metrics in this case.

        Returns
        -------
        """

        # implement NTA (salesforce)
        # this is the implementation from the iclr (and salesforce github) version, which mismatches the arxiv (and gluonnlp) version
        if not self.averaging_started and self.n > 0:
            if self.maximize:
                if len(self.val_logs) > self.n and average_trigger < max(
                    self.val_logs[: -self.n]
                ):
                    self.averaging_started = True
            else:
                if len(self.val_logs) > self.n and average_trigger > min(
                    self.val_logs[: -self.n]
                ):
                    self.averaging_started = True
            self.val_logs.append(average_trigger)


class NTA_V2(IterationAveragingStrategy):
    @validated()
    def __init__(self, n: int = 5, maximize: bool = False):
        r"""
        Parameters
        ----------
        n
            The non-monotone interval
        maximize
            Whether to maximize or minimize the validation metric
        val_logs
            Historical validation metrics
        """

        super().__init__()

        self.n = n
        self.maximize = maximize
        self.val_logs = []

    def update_average_trigger(self, average_trigger: Any):
        r"""
        Parameters
        ----------
        average_trigger
            The criteria to trigger averaging, evaluation metrics in this case.

        Returns
        -------
        """

        # implement NTA (gluonnlp)
        if not self.averaging_started and self.n > 0:
            if self.maximize:
                # in gluonnlp awd-lstm, "len(self.val_logs) > self.n" is used, but I think it should be ">=" instead
                if len(self.val_logs) >= self.n and average_trigger < max(
                    self.val_logs[-self.n :]
                ):
                    self.averaging_started = True
            else:
                if len(self.val_logs) >= self.n and average_trigger > min(
                    self.val_logs[-self.n :]
                ):
                    self.averaging_started = True
            self.val_logs.append(average_trigger)


class Alpha_Suffix(IterationAveragingStrategy):
    @validated()
    def __init__(self, epochs: int, alpha: float = 0.75):
        r"""
        Taking iteration average for the last epoch*alpha epochs

        Parameters
        ----------
        epochs
            The total number of epochs.
        alpha
            Proportion of averaging.
        alpha_suffix
            The epoch where iteration averaging starts.
        """

        super().__init__()

        assert alpha >= 0 and alpha <= 1

        self.alpha_suffix = epochs * (1.0 - alpha)

    def update_average_trigger(self, average_trigger: Any):
        r"""
        Parameters
        ----------
        average_trigger
            The current number of epoch.

        Returns
        -------
        """

        if not self.averaging_started:
            if average_trigger >= self.alpha_suffix:
                self.averaging_started = True

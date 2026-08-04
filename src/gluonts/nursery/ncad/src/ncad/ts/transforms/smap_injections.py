# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as np

from ncad.ts import TimeSeries
from ncad.ts import transforms as tr
from ncad.ts.transforms import TimeSeriesTransform


def normalise_values(values):

    values = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values)) * 2.0 - 1.0

    return values


def smap_injection(max_std=0.1):

    step_change_point_smap = SMAPChangePoint()

    transform = tr.Chain(
        [
            tr.ApplyWithProbability(SMAPSpikes(max_std=max_std), 0.4),
            SMAPChangePoint(max_std=max_std),
        ]
    )

    return transform


class SMAPSpikes(TimeSeriesTransform):
    def __init__(
        self,
        max_std=0.1,
    ):
        self.max_std = max_std

    def transform(
        self,
        ts: TimeSeries,
    ) -> TimeSeries:

        values = ts.values
        labels = ts.labels
        indicator = ts.indicator

        if np.nanstd(values) < self.max_std:

            method_choice = np.random.rand() > 0.7

            mask = np.zeros_like(values)

            if method_choice:
                mask += np.random.randn(len(values)) / 3.0

            # this is integers
            indices_of_swap = np.random.choice(
                range(len(values)), int(len(values) * (15.0 / 1000.0)), False
            )

            mask[indices_of_swap] = 1.0
            i = 1
            while np.random.rand() > 0.7:
                mask[np.clip(indices_of_swap + i, 0, len(values) - 1)] = 1.0
                i += 1

            values = normalise_values(values + mask)

        return TimeSeries(values, labels, indicator=indicator)


class SMAPChangePoint(TimeSeriesTransform):
    def __init__(
        self,
        max_additive_stepsize=50,
        min_stepsize_for_anomalous=5,
        max_change_duration=1000,
        min_change_duration=20,
        direction_options=("increase", "decrease"),
        number_anomalous_points_after_change=60 * 2,
        max_std=0.1,
    ):

        self.min_change_duration = min_change_duration
        self.max_additive_stepsize = max_additive_stepsize
        self.min_stepsize_for_anomalous = min_stepsize_for_anomalous
        self.max_change_duration = max_change_duration
        self.number_anomalous_points_after_change = number_anomalous_points_after_change
        self.direction_options = direction_options
        self.max_std = max_std

    def transform(self, ts: TimeSeries) -> TimeSeries:

        values = ts.values
        labels = ts.labels
        indicator = ts.indicator

        if np.nanstd(values) < self.max_std:
            LABEL_DELAY = 1

            step_size = np.random.rand() * self.max_additive_stepsize
            change_direction = 1 if np.random.choice(self.direction_options) == "increase" else -1

            step = step_size * change_direction

            new_magnitude = 4
            label_change = 1

            change_duration = np.random.randint(self.min_change_duration, self.max_change_duration)
            conv_mask = np.ones(change_duration) / change_duration

            change_location = np.random.randint(
                len(values) // 5, len(values) - self.number_anomalous_points_after_change
            )

            mask = np.ones(len(values) + len(conv_mask))
            mask[change_location:] = step

            mask = np.convolve(mask, conv_mask, mode="same")[: len(values)]

            labels_addition = [0] * change_location
            labels_addition += [label_change] * min(
                self.number_anomalous_points_after_change, len(values) - change_location
            )
            labels_addition += [0] * max(0, len(values) - len(labels_addition))
            labels_addition = np.concatenate(
                [np.zeros(LABEL_DELAY), np.array(labels_addition[:-LABEL_DELAY])]
            )

            values = normalise_values(values + mask)
            labels = np.logical_or(labels, labels_addition).astype(int)

            indicator[
                change_location : change_location + self.number_anomalous_points_after_change
            ] = 1

        return TimeSeries(values, labels, indicator=indicator)

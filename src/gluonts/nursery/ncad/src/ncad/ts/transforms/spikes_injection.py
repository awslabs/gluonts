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


from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from ncad.ts import TimeSeries
from ncad.ts import transforms as tr
from ncad.ts.transforms import TimeSeriesTransform
from ncad.ts.transforms.base import get_magnitude

RangeFloat = Tuple[float, float]
RangeInt = Tuple[int, Union[int, None]]


class LocalOutlier(TimeSeriesTransform):
    """
    Inject spikes based on local noise
    """

    def __init__(
        self,
        max_duration_spike: int = 2,
        spike_multiplier_range: RangeFloat = (0.5, 2.0),
        spike_value_range: RangeFloat = (-np.inf, np.inf),
        direction_options: List[str] = ["increase", "decrease"],
        area_radius: int = 100,
        num_spikes: int = 1,
    ) -> None:

        self.max_duration_spike = max_duration_spike
        self.spike_multiplier_range = spike_multiplier_range
        self.spike_value_range = spike_value_range
        self.direction_options = direction_options
        self.area_radius = area_radius
        self.num_spikes = num_spikes

    def transform(
        self,
        ts: TimeSeries,
        spike_location: Optional[np.ndarray] = None,
    ) -> TimeSeries:

        values = ts.values
        labels = ts.labels

        # Length and dimension of the TimeSeries
        T, ts_channels = ts.shape

        num_spikes = self.num_spikes
        if self.num_spikes < 1.0:
            num_spikes = int(self.num_spikes * T) + 1

        # sample a random magnitude for each injected spike
        # random uniform in spike_multiplier_range
        spike_multiplier = (
            np.random.rand(num_spikes, ts_channels)
            * (self.spike_multiplier_range[1] - self.spike_multiplier_range[0])
            + self.spike_multiplier_range[0]
        )
        # random sign
        sign = (
            2
            * (
                np.random.choice(self.direction_options, size=(num_spikes, ts_channels))
                == "increase"
            )
            - 1
        )
        spike_multiplier *= sign

        duration_spike = np.random.randint(low=1, high=self.max_duration_spike + 1, size=num_spikes)

        if spike_location is None:
            spike_location = np.random.randint(low=0, high=len(values) - duration_spike)
        else:
            spike_location = np.array(spike_location)
        assert spike_location.shape == (
            num_spikes,
        ), "The length of 'spike_location' must be equal to num_spikes"

        label_spike = 1

        local_range = np.zeros((num_spikes, 2, ts_channels))
        for i, t in enumerate(spike_location):
            local_left = np.maximum(0, t - self.area_radius)
            local_right = np.minimum(t + self.area_radius, len(values) - 1)

            area = values[local_left:local_right]
            local_range[i] = np.nanquantile(area, q=[0.05, 0.95], axis=0).reshape((2, ts_channels))
        spike_addition = (local_range[:, 1, :] - local_range[:, 0, :]) * spike_multiplier
        assert spike_addition.shape == (num_spikes, ts_channels)

        ## Set some of the spikes to zero, so that there is not always an anomaly in each dimension
        if ts_channels > 3:
            indices = np.random.choice(
                np.arange(spike_addition.size), replace=False, size=int(spike_addition.size * 0.35)
            )

            spike_addition[indices // ts_channels - 1, indices // num_spikes - 1] = 0

        ## Prepare the spike to be added
        add_spike = np.zeros_like(values)
        labels_addition = np.zeros_like(labels)
        for i, t in enumerate(spike_location):
            add_spike[t : t + duration_spike[i]] = spike_addition[i]
            labels_addition[t : t + duration_spike[i]] = label_spike

        values_out = values + np.clip(add_spike, *self.spike_value_range)
        labels_out = np.logical_or(labels, labels_addition).astype(int)

        ts_out = ts.copy()
        ts_out.values = values_out
        ts_out.labels = labels_out
        ts_out.indicator = labels_addition

        return ts_out


class SpikeSmoothed(TimeSeriesTransform):
    """
    Add smoothed spikes using a gaussian filter.
    """

    def __init__(
        self,
        duration_range: RangeInt = [20, 21],
        gaussian_filter_std=15,
        multiplier_range: RangeFloat = (5, 20),
        duration_before_labeled_anomaly=0,
        spike_value_range: RangeFloat = (-np.inf, np.inf),
        direction_options=("increase", "decrease"),
    ) -> None:

        self.duration_range = duration_range
        self.multiplier_range = multiplier_range
        self.gaussian_filter_std = gaussian_filter_std
        self.duration_before_labeled_anomaly = duration_before_labeled_anomaly
        self.spike_value_range = spike_value_range
        self.direction_options = direction_options

    def transform(self, ts: TimeSeries, spike_location: Optional[int] = None) -> TimeSeries:
        values = ts.values
        labels = ts.labels

        magnitude = get_magnitude(values)

        spike_multiplier = (np.random.rand() + self.multiplier_range[0]) * (
            self.multiplier_range[1] - self.multiplier_range[0]
        )

        duration_spike = np.random.randint(*self.duration_range)

        if spike_location is None:
            spike_location = np.random.randint(0, len(values) - duration_spike)
        label_change = 1
        if spike_location < self.duration_before_labeled_anomaly:
            label_change = 0

        spike_multiplier = (
            spike_multiplier
            if np.random.choice(self.direction_options) == "increase"
            else -spike_multiplier
        )

        std_for_smoothing = 0.05 + np.random.rand() * self.gaussian_filter_std
        # std_for_smoothing = np.random.rand() + self.gaussian_filter_std

        duration_spike += int(std_for_smoothing * 2)
        mult_spike = np.zeros_like(values)
        mult_spike[spike_location : spike_location + duration_spike] = spike_multiplier

        mult_spike = gaussian_filter(mult_spike, sigma=std_for_smoothing)

        std_adjustment_smoothing = self.gaussian_filter_std
        labels_addition = [0] * int(spike_location - std_adjustment_smoothing)
        labels_addition += [label_change] * min(
            duration_spike + int(2 * std_adjustment_smoothing), len(values) - spike_location
        )
        labels_addition += [0] * max(0, len(values) - len(labels_addition))

        # Alternative method, but not really working yet:
        # labels_addition = np.zeros_like(values)
        # labels_addition[spike_location:spike_location+duration_spike] += 1
        # labels_addition = gaussian_filter(labels_addition, sigma=self.gaussian_filter_std)
        # labels_addition = labels_addition > 0.2

        values = values + np.clip(mult_spike, *self.spike_value_range)
        labels = np.logical_or(labels, labels_addition).astype(int)

        indicator = ts.indicator
        indicator[spike_location : spike_location + duration_spike] = 1
        return TimeSeries(values, labels, indicator=indicator)

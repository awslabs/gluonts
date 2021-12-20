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


from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn

from pts.core.component import validated
from pts.dataset import InferenceDataLoader, DataEntry, FieldName
from pts.modules import DistributionOutput
from .forecast import Forecast, DistributionForecast, QuantileForecast, SampleForecast

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


def _extract_instances(x: Any) -> Any:
    """
    Helper function to extract individual instances from batched
    mxnet results.

    For a tensor `a`
      _extract_instances(a) -> [a[0], a[1], ...]

    For (nested) tuples of tensors `(a, (b, c))`
      _extract_instances((a, (b, c)) -> [(a[0], (b[0], c[0])), (a[1], (b[1], c[1])), ...]
    """
    if isinstance(x, (np.ndarray, torch.Tensor)):
        for i in range(x.shape[0]):
            # yield x[i: i + 1]
            yield x[i]
    elif isinstance(x, tuple):
        for m in zip(*[_extract_instances(y) for y in x]):
            yield tuple([r for r in m])
    elif isinstance(x, list):
        for m in zip(*[_extract_instances(y) for y in x]):
            yield [r for r in m]
    elif x is None:
        while True:
            yield None
    else:
        assert False


class ForecastGenerator(ABC):
    """
    Classes used to bring the output of a network into a class.
    """

    @abstractmethod
    def __call__(
        self,
        inference_data_loader: InferenceDataLoader,
        prediction_net: nn.Module,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        pass


class DistributionForecastGenerator(ForecastGenerator):
    def __init__(self, distr_output: DistributionOutput) -> None:
        self.distr_output = distr_output

    def __call__(
        self,
        inference_data_loader: InferenceDataLoader,
        prediction_net: nn.Module,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[DistributionForecast]:
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            outputs = prediction_net(*inputs)
            if output_transform is not None:
                outputs = output_transform(batch, outputs)

            distributions = [
                self.distr_output.distribution(*u) for u in _extract_instances(outputs)
            ]

            i = -1
            for i, distr in enumerate(distributions):
                yield DistributionForecast(
                    distr,
                    start_date=batch["forecast_start"][i],
                    freq=freq,
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                )
            assert i + 1 == len(batch["forecast_start"])


class QuantileForecastGenerator(ForecastGenerator):
    def __init__(self, quantiles: List[str]) -> None:
        self.quantiles = quantiles

    def __call__(
        self,
        inference_data_loader: InferenceDataLoader,
        prediction_net: nn.Module,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            outputs = prediction_net(*inputs).cpu().numpy()
            if output_transform is not None:
                outputs = output_transform(batch, outputs)

            i = -1
            for i, output in enumerate(outputs):
                yield QuantileForecast(
                    output,
                    start_date=batch["forecast_start"][i],
                    freq=freq,
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                    forecast_keys=self.quantiles,
                )
            assert i + 1 == len(batch["forecast_start"])


class SampleForecastGenerator(ForecastGenerator):

    @validated()
    def __init__(self):
        pass

    def __call__(
        self,
        inference_data_loader: InferenceDataLoader,
        prediction_net: nn.Module,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            outputs = prediction_net(*inputs).cpu().numpy()
            if output_transform is not None:
                outputs = output_transform(batch, outputs)
            if num_samples:
                num_collected_samples = outputs[0].shape[0]
                collected_samples = [outputs]
                while num_collected_samples < num_samples:
                    outputs = prediction_net(*inputs).cpu().numpy()
                    if output_transform is not None:
                        outputs = output_transform(batch, outputs)
                    collected_samples.append(outputs)
                    num_collected_samples += outputs[0].shape[0]
                outputs = [
                    np.concatenate(s)[:num_samples] for s in zip(*collected_samples)
                ]
                assert len(outputs[0]) == num_samples
            i = -1
            for i, output in enumerate(outputs):
                yield SampleForecast(
                    output,
                    start_date=batch["forecast_start"][i],
                    freq=freq,
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                )
            assert i + 1 == len(batch["forecast_start"])

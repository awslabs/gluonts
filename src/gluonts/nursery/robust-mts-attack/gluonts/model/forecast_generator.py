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

import logging
from functools import singledispatch
from typing import Callable, Iterator, List, Optional

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataLoader
from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast

logger = logging.getLogger(__name__)

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]

LOG_CACHE = set([])
OUTPUT_TRANSFORM_NOT_SUPPORTED_MSG = (
    "The `output_transform` argument is not supported and will be ignored."
)
NOT_SAMPLE_BASED_MSG = "Forecast is not sample based. Ignoring parameter `num_samples` from predict method."


def log_once(msg):
    global LOG_CACHE
    if msg not in LOG_CACHE:
        logger.info(msg)
        LOG_CACHE.add(msg)


# different deep learning frameworks generate predictions and the tensor to numpy conversion differently,
# use a dispatching function to prevent needing a ForecastGenerators for each framework
@singledispatch
def predict_to_numpy(prediction_net, tensor) -> np.ndarray:
    raise NotImplementedError


@singledispatch
def recursively_zip_arrays(x) -> Iterator:
    """
    Helper function to recursively zip nested collections of arrays.

    This defines the fallback implementation, which one can specialized for specific types
    using by doing ``@recursively_zip_arrays.register`` on the type. Implementations for
    lists, tuples, and NumPy arrays are provided.

    For an array `a` (e.g. a numpy array)

        _extract_instances(a) -> [a[0], a[1], ...]

    For (nested) tuples of arrays `(a, (b, c))`

        _extract_instances((a, (b, c)) -> [(a[0], (b[0], c[0])), (a[1], (b[1], c[1])), ...]
    """
    raise NotImplementedError


@recursively_zip_arrays.register(np.ndarray)
def _(x: np.ndarray) -> Iterator[list]:
    for i in range(x.shape[0]):
        yield x[i]


@recursively_zip_arrays.register(tuple)
def _(x: tuple) -> Iterator[tuple]:
    for m in zip(*[recursively_zip_arrays(y) for y in x]):
        yield tuple([r for r in m])


@recursively_zip_arrays.register(list)
def _(x: list) -> Iterator[list]:
    for m in zip(*[recursively_zip_arrays(y) for y in x]):
        yield [r for r in m]


@recursively_zip_arrays.register(type(None))
def _(x: type(None)) -> Iterator[type(None)]:
    while True:
        yield None


@singledispatch
def make_distribution_forecast(distr, *args, **kwargs) -> Forecast:
    raise NotImplementedError


class ForecastGenerator:
    """
    Classes used to bring the output of a network into a class.
    """

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        raise NotImplementedError()


class QuantileForecastGenerator(ForecastGenerator):
    @validated()
    def __init__(self, quantiles: List[str]) -> None:
        self.quantiles = quantiles

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            outputs = predict_to_numpy(prediction_net, inputs)
            if output_transform is not None:
                outputs = output_transform(batch, outputs)

            if num_samples:
                log_once(NOT_SAMPLE_BASED_MSG)

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
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            outputs = predict_to_numpy(prediction_net, inputs)
            if output_transform is not None:
                outputs = output_transform(batch, outputs)
            if num_samples:
                num_collected_samples = outputs[0].shape[0]
                collected_samples = [outputs]
                while num_collected_samples < num_samples:
                    outputs = predict_to_numpy(prediction_net, inputs)
                    if output_transform is not None:
                        outputs = output_transform(batch, outputs)
                    collected_samples.append(outputs)
                    num_collected_samples += outputs[0].shape[0]
                outputs = [
                    np.concatenate(s)[:num_samples]
                    for s in zip(*collected_samples)
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


class DistributionForecastGenerator(ForecastGenerator):
    @validated()
    def __init__(self, distr_output) -> None:
        self.distr_output = distr_output

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            outputs = prediction_net(*inputs)

            if output_transform:
                log_once(OUTPUT_TRANSFORM_NOT_SUPPORTED_MSG)
            if num_samples:
                log_once(NOT_SAMPLE_BASED_MSG)

            distributions = [
                self.distr_output.distribution(*u)
                for u in recursively_zip_arrays(outputs)
            ]

            i = -1
            for i, distr in enumerate(distributions):
                yield make_distribution_forecast(
                    distr,
                    start_date=batch["forecast_start"][i],
                    freq=freq,
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                )
            assert i + 1 == len(batch["forecast_start"])

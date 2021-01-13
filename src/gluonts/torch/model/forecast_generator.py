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

from typing import Any, List, Optional, Iterator

from torch import nn
import torch
import numpy as np

from gluonts.dataset.loader import DataLoader
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast_generator import ForecastGenerator, OutputTransform
from gluonts.torch.model.forecast import DistributionForecast
from gluonts.torch.modules.distribution_output import DistributionOutput


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


class DistributionForecastGenerator(ForecastGenerator):
    def __init__(self, distr_output: DistributionOutput) -> None:
        self.distr_output = distr_output

    def __call__(
        self,
        inference_data_loader: DataLoader,
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
                self.distr_output.distribution(*u)
                for u in _extract_instances(outputs)
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

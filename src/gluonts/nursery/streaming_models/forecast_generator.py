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
from typing import Any, Iterator, List, Optional

import mxnet as mx
import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataLoader, InferenceDataLoader
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    OutputTransform,
    recursively_zip_arrays,
)
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.model.forecast import DistributionForecast

from .predictor import NETWORK_STATE_KEY, PREDICTOR_STATE_KEY

logger = logging.getLogger(__name__)


def to_numpy(x: Any) -> Any:
    if isinstance(x, mx.nd.NDArray):
        return x.asnumpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (float, int, np.float)):
            return x
        else:
            tmp = [to_numpy(xi) for xi in x]
            if isinstance(x, tuple):
                return tuple(tmp)
            return tmp
    return x


class StatefulDistributionForecastGenerator(ForecastGenerator):
    @validated()
    def __init__(self, distr_output: DistributionOutput) -> None:
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
    ) -> Iterator[DistributionForecast]:
        if output_transform is not None:
            logger.info("The `output_transform` argument will be ignored.")
        if num_samples is not None:
            logger.info(
                "Forecast is not sample based. Ignoring parameter `num_samples`."
            )
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            transformation_states = {
                k: b for k, b in batch.items() if k.startswith("s:")
            }
            outputs = prediction_net(*inputs)
            distr_args_scale_loc = outputs[:-1]
            network_state_batch = outputs[-1]

            assert len(distr_args_scale_loc) == 3

            distributions = [
                self.distr_output.distribution(*u)
                for u in recursively_zip_arrays(distr_args_scale_loc)
            ]

            network_states = [
                s for s in recursively_zip_arrays(network_state_batch)
            ]

            idx = -1
            for idx, (distr, network_state) in enumerate(
                zip(distributions, network_states)
            ):
                trans_states = {
                    k: to_numpy(b[idx])
                    for k, b in transformation_states.items()
                }
                yield DistributionForecast(
                    distr,
                    start_date=batch["forecast_start"][idx],
                    freq=freq,
                    item_id=batch[FieldName.ITEM_ID][idx]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info={
                        PREDICTOR_STATE_KEY: {
                            NETWORK_STATE_KEY: to_numpy(network_state),
                            **trans_states,
                        }
                    },
                )
            assert idx + 1 == len(batch["forecast_start"])

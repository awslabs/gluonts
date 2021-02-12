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

from copy import deepcopy
from pathlib import Path
from pydoc import locate
from typing import Any, Callable, Dict, List, Optional, Tuple

import mxnet as mx
import numpy as np
import pandas as pd

from gluonts.core.component import fqname_for, validated
from gluonts.core.serde import dump_json, load_json
from gluonts.dataset.common import DataEntry
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from gluonts.mx.context import get_mxnet_context
from gluonts.mx.model.forecast import DistributionForecast

PREDICTOR_STATE_KEY = "predictor_state"
NETWORK_STATE_KEY = "network_state"

StreamState = Dict[str, Any]


class StreamPredictor:
    """
    An abstract type for predictors that handle streams of data and explicit state updates.
    """

    @property
    def lead_time(self) -> int:
        raise NotImplementedError()

    def initial_state(self) -> StreamState:
        raise NotImplementedError()

    def step(
        self, data: DataEntry, state: StreamState
    ) -> Tuple[Optional[Forecast], StreamState]:
        raise NotImplementedError()

    def serialize(self, path: Path) -> None:
        with (path / "type.txt").open("w") as fp:
            fp.write(fqname_for(self.__class__))

    @classmethod
    def deserialize(cls, path: Path, *args, **kwargs) -> "StreamPredictor":
        with (path / "type.txt").open("r") as fp:
            tpe_str = fp.readline()
            tpe = locate(tpe_str)

        assert isinstance(tpe, type)
        assert issubclass(tpe, StreamPredictor)

        return tpe.deserialize(path, *args, **kwargs)  # type: ignore


class StateAwarePredictorWrapper(StreamPredictor):
    """
    A wrapper around GluonTS predictors that are "state aware", i.e. that
    consume a state field when processing a data entry, and include the
    update state in the returned forecasts (in the "info" field).

    Parameters
    ----------
    predictor
        The GluonTS predictor object to be wrapped.
    state_initializer
        A function with no argument, that returns the initial state for the predictor.
    """

    @validated()
    def __init__(
        self,
        predictor: Predictor,
        state_initializer: Callable[[], Any],
    ) -> None:
        # TODO: this should rather be undefined (None? +infty?) because there's
        # TODO: no concept of "prediction" length in the streaming case
        assert predictor.prediction_length == 1

        self.predictor = predictor
        self.state_initializer = state_initializer

    @property
    def lead_time(self) -> int:
        return self.predictor.lead_time

    def initial_state(self) -> StreamState:
        return self.state_initializer()

    def step(
        self, data: DataEntry, state: StreamState
    ) -> Tuple[DistributionForecast, StreamState]:
        data_and_state = {**data, **state}

        # TODO: the following two lines could be nicer, if only predictors
        # TODO: allowed for a 1-item prediction directly
        forecasts = self.predictor.predict([data_and_state])
        forecast = next(iter(forecasts))

        assert isinstance(forecast, DistributionForecast)

        return forecast, forecast.info[PREDICTOR_STATE_KEY]

    def serialize(self, path: Path) -> None:
        (path / "predictor").mkdir(parents=True, exist_ok=True)
        self.predictor.serialize(path / "predictor")

        super().serialize(path)
        with (path / "wrapper_parameters.json").open("w") as fp:
            parameters = dict(
                state_initializer=self.state_initializer,
            )
            print(dump_json(parameters), file=fp)

    @classmethod
    def deserialize(
        cls,
        path: Path,
        ctx: Optional[mx.Context] = None,
        *args,
        **kwargs,
    ) -> "StateAwarePredictorWrapper":
        ctx = ctx if ctx is not None else get_mxnet_context()
        with mx.Context(ctx):
            predictor = Predictor.deserialize(path / "predictor")  # type: ignore
            with (path / "wrapper_parameters.json").open("r") as fp:
                parameters = load_json(fp.read())
            return StateAwarePredictorWrapper(
                predictor=predictor, **parameters
            )

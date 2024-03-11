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

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mxnet as mx
import pandas as pd
from pydantic import BaseModel
from toolz import tail

from gluonts.dataset.loader import as_stacked_batches
from gluonts.model import Predictor, SampleForecast
from gluonts.time_feature import (
    day_of_month,
    day_of_week,
    day_of_year,
    hour_of_day,
    minute_of_hour,
    month_of_year,
    second_of_minute,
    week_of_year,
)


feature_by_name = {
    "day_of_month": day_of_month,
    "day_of_week": day_of_week,
    "day_of_year": day_of_year,
    "hour_of_day": hour_of_day,
    "minute_of_hour": minute_of_hour,
    "month_of_year": month_of_year,
    "second_of_minute": second_of_minute,
    "week_of_year": week_of_year,
}


class DataShape(BaseModel):
    shape: Tuple[int, ...]
    name: str


class DeepARConfig(BaseModel):
    cardinality: List[int]
    category_provided: bool
    data_shapes: List[DataShape]

    num_dynamic_feat: int
    train_length: int
    prediction_length: int
    time_freq: str

    date_feature_names: List[str]

    def time_features(self):
        return TimeFeatures(
            [feature_by_name[name] for name in self.date_feature_names],
            freq=self.time_freq,
            past_length=self.train_length,
            future_length=self.prediction_length,
            num_dynamic_feat=self.num_dynamic_feat,
        )


@dataclass
class TimeFeatures:
    features: list
    freq: str
    past_length: int
    future_length: int
    num_dynamic_feat: int

    def __call__(self, entry):
        periods = pd.period_range(
            start=entry["start"],
            freq=self.freq,
            periods=self.past_length + self.future_length,
        )

        past = []
        future = []

        for feature in self.features:
            feat = feature(periods)
            past.append(feat[: self.past_length])
            future.append(feat[self.past_length :])

        age = np.log10(2 + np.arange(self.past_length + self.future_length))

        past.append(age[: self.past_length])
        future.append(age[self.past_length :])

        if self.num_dynamic_feat:
            raise NotImplementedError

        return {
            "timeFeaturesTrain": np.vstack(past),
            "timeFeaturesPred": np.vstack(future),
        }


@dataclass
class SageMakerDeepARPredictor(Predictor):
    module: mx.module.Module
    config: DeepARConfig

    @classmethod
    def deserialize(cls, path: Path, **kwargs) -> "Predictor":
        symbol = list(path.glob("model*-symbol.json"))
        assert len(symbol) == 1
        symbol = symbol[0]

        model_name = symbol.name.rsplit("-", 1)[0]

        config = DeepARConfig.parse_file(path / f"{model_name}-config.json")

        module = mx.module.Module.load(
            str(path / model_name),
            epoch=0,
            data_names=[data_shape.name for data_shape in config.data_shapes],
            label_names=None,
        )

        data_shapes = [
            mx.io.DataDesc(name=desc.name, shape=desc.shape)
            for desc in config.data_shapes
        ]
        module.bind(
            data_shapes=data_shapes, label_shapes=None, for_training=False
        )

        return SageMakerDeepARPredictor(module=module, config=config)

    def item_features(self, entry):
        if self.config.category_provided:
            return entry["feat_static_cat"]

        return [0]

    def pipeline(self, entry):
        time_feat = self.config.time_features()

        target = tail(self.config.train_length, entry["target"])
        to_pad = len(target) - self.config.train_length
        target = np.concatenate([np.zeros(to_pad), target])

        observed_indicator = np.invert(np.isnan(target)).astype(np.float32)
        is_padded = np.concatenate(
            [np.ones(to_pad), np.zeros(self.config.train_length - to_pad)]
        )

        return {
            "trainTarget": target,
            **time_feat(entry),
            "itemFeatures": [0],
            "observedValuesTrain": observed_indicator,
            "isPaddedTrain": is_padded,
        }

    def predict(self, data, batch_size=32):
        data1, data2 = itertools.tee(data)

        inputs = map(self.pipeline, data1)

        for batch in as_stacked_batches(
            inputs, batch_size=batch_size, output_type=mx.nd.array
        ):
            batch = mx.io.DataBatch(list(batch.values()))

            self.module.forward(batch, is_train=False)
            samples_batch = self.module.get_outputs()[0]

            for samples in samples_batch.asnumpy():
                input_ = next(data2)
                start = pd.Period(input_["start"], freq=self.config.time_freq)

                yield SampleForecast(
                    samples=samples,
                    start_date=start + len(input_["target"]),
                )

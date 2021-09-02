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


from itertools import islice
from pathlib import Path
import os


from gluonts.dataset.repository.datasets import get_dataset
from gluonts.nursery.autogluon_tabular import TabularEstimator
from gluonts.model.predictor import Predictor


def run_example():
    dataset = get_dataset("electricity")
    serialize_path = Path("GluonTSTabularPredictor")
    estimator = TabularEstimator(
        freq="H",
        prediction_length=24,
        time_limit=600,  # ten minutes for training
        disable_auto_regression=True,  # makes prediction faster, but potentially less accurate
        last_k_for_val=24,  # split the last 24 targets from each time series to be the validation data
        quantiles_to_predict=[0.1, 0.5, 0.9],
    )

    n_train = 5

    training_data = list(islice(dataset.train, n_train))

    predictor = estimator.train(training_data=training_data)

    os.makedirs(serialize_path, exist_ok=True)
    predictor.serialize(serialize_path)
    predictor = None
    # the quantiles_to_predict parameters should be List[str] type
    predictor = Predictor.deserialize(serialize_path)
    forecasts = list(predictor.predict(training_data))
    print(forecasts)


if __name__ == "__main__":
    run_example()

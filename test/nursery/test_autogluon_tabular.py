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


import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.nursery.autogluon_tabular import LocalTabularPredictor


def test_autogluon_tabular():
    # create a dataset
    dataset = ListDataset(
        [
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [1089.2, 1078.91, 1099.88, 35790.55, 34096.95, 34906.95],
                ),
            },
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [1099.2, 1098.91, 1069.88, 35990.55, 34076.95, 34766.95],
                ),
            },
        ],
        freq="W-SUN",
    )
    prediction_length = 2
    freq = "W-SUN"
    predictor = LocalTabularPredictor(
        freq=freq,
        prediction_length=prediction_length,
    )
    forecasts_it = predictor.predict(dataset)
    forecasts = list(forecasts_it)

    for entry, forecast in zip(dataset, forecasts):
        ts = to_pandas(entry)
        start_timestamp = ts.index[-1] + pd.tseries.frequencies.to_offset(freq)
        assert forecast.samples.shape[1] == prediction_length
        assert forecast.start_date == start_timestamp

    return forecasts


if __name__ == "__main__":
    print(test_autogluon_tabular())

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

# Third-party imports
import numpy as np
import pandas as pd
import os
from pathlib import Path

# First-party imports
from gluonts.model.naive_2 import naive_2


# DISCLAIMER:
# this script is only to test whether the R implementation
# of naive_2 produces the same outputs as out python implementation

R_INPUT_FILE = "r_naive_2_inputs.csv"
R_OUTPUT_FILE = "r_naive_2_outputs.csv"


def load_naive_2_data():
    test_directory_path = Path(os.getenv("PYTEST_CURRENT_TEST")).parents[0]
    r_naive_2_inputs = pd.read_csv(test_directory_path / R_INPUT_FILE).values
    r_naive_2_outputs = pd.read_csv(test_directory_path / R_OUTPUT_FILE).values
    return r_naive_2_inputs, r_naive_2_outputs


# To generate the above dataset use the following script:
# https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
# and then paste the following code in as the first line:
"""
set.seed(1234567)
"""
# an the below code at the very end:
"""
for (i in 1:length(data_train)){
   insample <- data_train[[i]]
   d <- insample[1:14]
   v <-paste(d, collapse=", ")
   cat(c("[",v,"],"))
}

print('\n')

for (i in 1:length(data_train)){
  
  insample <- data_train[[i]]
  forecasts <- Benchmarks(input=insample, fh=fh)

  f <- forecasts[[1]][1:6]
  v <-paste(f, collapse=", ")
  cat(c("[",v,"],"))

}
"""

# R script variables:
FH = 6  # The forecasting horizon examined
FRQ = 1  # The frequency of the data


def test_naive_2(prediction_length=FH, season_length=FRQ):
    r_naive_2_inputs, r_naive_2_outputs = load_naive_2_data()
    predictions = []
    for i in range(len(r_naive_2_inputs)):
        predictions.append(
            naive_2(
                r_naive_2_inputs[i],
                prediction_length=prediction_length,
                season_length=season_length,
            )
        )
    predictions = np.array(predictions)

    assert np.allclose(r_naive_2_outputs, predictions)

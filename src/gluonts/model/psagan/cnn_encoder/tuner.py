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

import time
from itertools import product

from gluonts.model.psagan.cnn_encoder.run_CausalCNN_sagemaker import run

hyperparameters = {
    "num_epochs": [600],
    "lr": [0.0001],
    "save_display_frq": [1],
    "batch_size": [32],
    "nb_features": [1],
    "nb_channels": [40],
    "depth": [10],
    "reduced_size": [160],
    "size_embedding": [80],
    "kernel_size": [3],
    "subseries_length": [30],  # Not used
    "context_length": [200],  # Not used
    "max_len": [300],
    "nb_negative_samples": [20],
    "num_workers": [0],
    "dataset_name": [
        "traffic",
    ],
    "scaling": ["global"],
    "device": ["gpu"],
}


def grid_search(hyper):

    combos = product(
        *([(key, val) for val in value] for key, value in hyper.items())
    )
    combos = [dict(combination) for combination in combos]
    for combo in combos:
        time.sleep(3)
        run(**combo)


grid_search(hyperparameters)

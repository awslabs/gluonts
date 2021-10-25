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

from itertools import product
from math import log2

from gluonts.model.psagan.helpers import (
    cardinality_fct,
    ds,
    nb_features_fct,
    set_context_length,
)
from gluonts.model.psagan.run_synthetic_sagemaker import run

hyperparameters = {
    "num_epochs": [2500],
    "lr_generator": [0.00005],
    "lr_discriminator": [0.00005],
    "betas_generator": [(0.9, 0.999)],
    "betas_discriminator": [(0.9, 0.999)],
    "schedule": [[1000]],
    "nb_step_discrim": [1],
    "nb_epoch_fade_in_new_layer": [500],
    "EMA_value": [0.2],
    "batch_size": [128],
    "num_batches_per_epoch": [10],
    "target_len": [2 ** 4],
    "ks_conv": [3],
    "key_features": [1],
    "value_features": [1],
    "ks_value": [1],
    "ks_query": [1],
    "ks_key": [1],
    "ds": [],
    "num_workers": [4],
    "device": ["gpu"],
    "use_loss": ["lsgan"],
    "momment_loss": [0.1],
    "scaling_penalty": [0],
    "path_to_pretrain": [False],
    "scaling": ["NoScale"],
    "cardinality": [[None]],
    "embedding_dim": [10],
    "self_attention": [],
    "channel_nb": [32],
    "encoder_network_factor": [0],
    "pos_enc_dimension": [0],
    "context_length": [10],
    "LARS": [False],
    "missing_values_stretch": [0],
    "path_to_csv": [
        "./gluonts/model/syntheticTransformer/TrainingFollowUp.csv"
    ],
    "cold_start_value": [None],
    # "path_to_pretrain" : ["./gluonts/model/syntheticTransformer/pretrained_model/synthetic_model_24_03_2021__17_14_07/checkpoint_pretrain_1990.pt"]
}


nb_features_dataset = {
    "m4_hourly": [4],
    "electricity": [4],
    "solar-energy": [5],
    "exchange_rate": [3],
    "traffic": [4],
    "s3://s3-violation-test-bucket/datasets/solar_nips_scaled/": [4],
    "s3://s3-violation-test-bucket/datasets/traffic_nips_scaled/": [4],
    "s3://s3-violation-test-bucket/datasets/m4_hourly_scaled/": [4],
    "s3://s3-violation-test-bucket/datasets/exchange_rate_nips_scaled/": [3],
    "s3://s3-violation-test-bucket/datasets/electricity_nips_scaled/": [4],
}


target_len = [256]  # , 1024]
# target_len = [16, 32, 64, 128, 256]  # , 512, 1024]
# epochs = [2500, 3500, 4500, 5500, 6500]  # , 7500, 8500]
epochs = [6500]  # , 8500]
schedule = [
    # [1000],
    # [1000, 2000],
    # [1000, 2000, 3000],
    # [1000, 2000, 3000, 4000],
    [1000, 2000, 3000, 4000, 5000],
    # [1000, 2000, 3000, 4000, 5000, 6000],
    # [1000, 2000, 3000, 4000, 5000, 6000, 7000],
]


def grid_search(hyper):
    combos = product(
        *([(key, val) for val in value] for key, value in hyper.items())
    )
    combos = [dict(combination) for combination in combos]
    list_of_combos = []
    for combo in combos:
        list_of_combos.append(combo.items)
    textfile = open("a_file.txt", "w")
    for element in list_of_combos:
        textfile.write(element + "\n")
    textfile.close()
    run(**combo)


hyperparameters["nb_epoch_fade_in_new_layer"] = [500]
hyperparameters["LARS"] = [False]
hyperparameters["num_batches_per_epoch"] = [100]
hyperparameters["batch_size"] = [32, 64, 128, 256, 512]
hyperparameters["pos_enc_dimension"] = [0]
hyperparameters["context_length"] = [0]
hyperparameters["channel_nb"] = [32]
hyperparameters["momment_loss"] = [1]
hyperparameters["use_loss"] = ["lsgan"]
hyperparameters["self_attention"] = [True]
hyperparameters["encoder_network_factor"] = [0]


for nb_runs in range(1):
    for context_length in [0]:
        CTX_LEN = context_length
        hyperparameters["context_length"] = [context_length]
        for missing_value in [
            0,
            # 5,
            # 50,
            # 110,
        ]:  # Acceptable values: [5, 10, 20, 35, 50, 65, 85, 110]
            assert missing_value in [
                0,
                5,
                10,
                20,
                35,
                50,
                65,
                85,
                110,
            ]
            hyperparameters["missing_values_stretch"] = [missing_value]
            for cold_start_value in [0]:
                assert cold_start_value in [
                    0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ]
                if cold_start_value > 0:
                    assert missing_value == 0
                if missing_value > 0:
                    assert cold_start_value == 0
                hyperparameters["cold_start_value"] = [cold_start_value]
                for d in ds(missing_value, cold_start_value):
                    hyperparameters["ds"] = [d]
                    nb_features_dataset = nb_features_fct(
                        missing_value, cold_start_value
                    )
                    hyperparameters["nb_features"] = [
                        nb_features_dataset[hyperparameters["ds"][0]][0]
                        + hyperparameters["embedding_dim"][0]
                        + 1  # Because of the AddAgeFeatures
                        + hyperparameters["pos_enc_dimension"][
                            0
                        ]  # Because of the positional encoding
                    ]
                    cardinality_dict = cardinality_fct(
                        missing_value, cold_start_value
                    )
                    hyperparameters["cardinality"] = [
                        [cardinality_dict[hyperparameters["ds"][0]]]
                    ]
                    for elmnt in zip(target_len, epochs, schedule,):
                        hyperparameters["target_len"][0] = elmnt[0]
                        hyperparameters["num_epochs"][0] = elmnt[1]
                        hyperparameters["schedule"][0] = elmnt[2]
                        for l in hyperparameters["schedule"]:
                            assert (
                                len(l)
                                == log2(hyperparameters["target_len"][0]) - 3
                            )
                            hyperparameters = set_context_length(
                                hyperparameters, CTX_LEN
                            )
                            grid_search(hyperparameters)

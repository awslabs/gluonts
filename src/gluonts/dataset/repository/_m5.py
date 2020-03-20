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

from pathlib import Path
import os
import json

import pandas as pd
import numpy as np

from gluonts.dataset.repository._util import metadata, save_to_file
from gluonts.dataset.field_names import FieldName


def generate_m5_dataset(
    dataset_path: Path, pandas_freq: str, prediction_length: int
):
    cal_path = f"{dataset_path}/calendar.csv"
    sales_path = f"{dataset_path}/sales_train_validation.csv"

    if not os.path.exists(cal_path) or not os.path.exists(sales_path):
        raise RuntimeError(
            f"M5 data is available on Kaggle (https://www.kaggle.com/c/m5-forecasting-accuracy/data). "
            f"You first need to agree to the terms of the competition before being able to download the data. "
            f"After you have done that, please supply the files at {dataset_path}."
        )

    # Read M5 data from dataset_path
    calendar = pd.read_csv(cal_path)
    sales_train_validation = pd.read_csv(sales_path)
    submission_prediction_length = prediction_length * 2

    # Build dynamic features
    cal_features = calendar.drop(
        [
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_name_2",
            "d",
        ],
        axis=1,
    )
    cal_features["event_type_1"] = cal_features["event_type_1"].apply(
        lambda x: 0 if str(x) == "nan" else 1
    )
    cal_features["event_type_2"] = cal_features["event_type_2"].apply(
        lambda x: 0 if str(x) == "nan" else 1
    )
    test_cal_features = cal_features.values.T
    train_cal_features = test_cal_features[
        :, : -submission_prediction_length - prediction_length
    ]
    test_cal_features = test_cal_features[:, :-submission_prediction_length]

    test_cal_features_list = [test_cal_features] * len(sales_train_validation)
    train_cal_features_list = [train_cal_features] * len(
        sales_train_validation
    )

    # Build static features
    state_ids = (
        sales_train_validation["state_id"].astype("category").cat.codes.values
    )
    state_ids_un = np.unique(state_ids)
    store_ids = (
        sales_train_validation["store_id"].astype("category").cat.codes.values
    )
    store_ids_un = np.unique(store_ids)
    cat_ids = (
        sales_train_validation["cat_id"].astype("category").cat.codes.values
    )
    cat_ids_un = np.unique(cat_ids)
    dept_ids = (
        sales_train_validation["dept_id"].astype("category").cat.codes.values
    )
    dept_ids_un = np.unique(dept_ids)
    item_ids = (
        sales_train_validation["item_id"].astype("category").cat.codes.values
    )
    item_ids_un = np.unique(item_ids)
    stat_cat_list = [item_ids, dept_ids, cat_ids, store_ids, state_ids]
    stat_cat = np.concatenate(stat_cat_list)
    stat_cat = stat_cat.reshape(len(stat_cat_list), len(item_ids)).T
    cardinalities = [
        len(item_ids_un),
        len(dept_ids_un),
        len(cat_ids_un),
        len(store_ids_un),
        len(state_ids_un),
    ]

    # Build target series
    train_df = sales_train_validation.drop(
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1
    )
    test_target_values = train_df.values.copy()
    train_target_values = [ts[:-prediction_length] for ts in train_df.values]
    dates = ["2011-01-29 00:00:00" for _ in range(len(sales_train_validation))]

    # Create metadata file
    meta_file = dataset_path / "metadata.json"
    with open(meta_file, "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=cardinalities,
                    freq=pandas_freq,
                    prediction_length=prediction_length,
                )
            )
        )

    # Build training set
    train_file = dataset_path / "train" / "data.json"
    train_ds = [
        {
            FieldName.TARGET: target.tolist(),
            FieldName.START: start,
            FieldName.FEAT_DYNAMIC_REAL: fdr.tolist(),
            FieldName.FEAT_STATIC_CAT: fsc.tolist(),
        }
        for (target, start, fdr, fsc) in zip(
            train_target_values, dates, train_cal_features_list, stat_cat
        )
    ]
    save_to_file(train_file, train_ds)

    # Build testing set
    test_file = dataset_path / "test" / "data.json"
    test_ds = [
        {
            FieldName.TARGET: target.tolist(),
            FieldName.START: start,
            FieldName.FEAT_DYNAMIC_REAL: fdr.tolist(),
            FieldName.FEAT_STATIC_CAT: fsc.tolist(),
        }
        for (target, start, fdr, fsc) in zip(
            test_target_values, dates, test_cal_features_list, stat_cat
        )
    ]
    save_to_file(test_file, test_ds)

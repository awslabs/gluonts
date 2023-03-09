import json
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._util import metadata, save_to_file
from gluonts.time_feature.holiday import squared_exponential_kernel
from pts.feature import CustomDateFeatureSet


def generate_pts_m5_dataset(
    dataset_path: Path,
    pandas_freq: str,
    prediction_length: int = 28,
    alpha: float = 0.5,
):
    cal_path = f"{dataset_path}/calendar.csv"
    sales_path = f"{dataset_path}/sales_train_validation.csv"
    sales_test_path = f"{dataset_path}/sales_train_evaluation.csv"
    sell_prices_path = f"{dataset_path}/sell_prices.csv"

    if not os.path.exists(cal_path) or not os.path.exists(sales_path):
        raise RuntimeError(
            f"M5 data is available on Kaggle (https://www.kaggle.com/c/m5-forecasting-accuracy/data). "
            f"You first need to agree to the terms of the competition before being able to download the data. "
            f"After you have done that, please copy the files into {dataset_path}."
        )

    # Read M5 data from dataset_path
    calendar = pd.read_csv(cal_path, parse_dates=True)
    calendar.sort_index(inplace=True)
    calendar.date = pd.to_datetime(calendar.date)

    sales_train_validation = pd.read_csv(
        sales_path,
        index_col=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )
    sales_train_validation.sort_index(inplace=True)

    sales_train_evaluation = pd.read_csv(
        sales_test_path,
        index_col=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )
    sales_train_evaluation.sort_index(inplace=True)

    sell_prices = pd.read_csv(sell_prices_path, index_col=["item_id", "store_id"])
    sell_prices.sort_index(inplace=True)

    @lru_cache(maxsize=None)
    def get_sell_price(item_id, store_id):
        return calendar.merge(
            sell_prices.loc[item_id, store_id], on=["wm_yr_wk"], how="left"
        ).sell_price

    # Build dynamic features
    kernel = squared_exponential_kernel(alpha=alpha)
    event_1 = CustomDateFeatureSet(calendar[calendar.event_name_1.notna()].date, kernel)
    event_2 = CustomDateFeatureSet(calendar[calendar.event_name_2.notna()].date, kernel)

    snap_CA = CustomDateFeatureSet(calendar[calendar.snap_CA == 1].date, kernel)
    snap_TX = CustomDateFeatureSet(calendar[calendar.snap_TX == 1].date, kernel)
    snap_WI = CustomDateFeatureSet(calendar[calendar.snap_WI == 1].date, kernel)

    time_index = pd.to_datetime(calendar.date)
    event_1_feature = event_1(time_index)
    event_2_feature = event_2(time_index)

    snap_CA_feature = snap_CA(time_index)
    snap_TX_feature = snap_TX(time_index)
    snap_WI_feature = snap_WI(time_index)

    # Build static features
    sales_train_validation["state"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(5)
    ).codes
    sales_train_validation["store"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(4)
    ).codes
    sales_train_validation["cat"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(3)
    ).codes
    sales_train_validation["dept"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(2)
    ).codes
    sales_train_validation["item"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(1)
    ).codes

    sales_train_evaluation["state"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(5)
    ).codes
    sales_train_evaluation["store"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(4)
    ).codes
    sales_train_evaluation["cat"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(3)
    ).codes
    sales_train_evaluation["dept"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(2)
    ).codes
    sales_train_evaluation["item"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(1)
    ).codes

    feat_static_cat = [
        {
            "name": "state_id",
            "cardinality": len(sales_train_validation["state"].unique()),
        },
        {
            "name": "store_id",
            "cardinality": len(sales_train_validation["store"].unique()),
        },
        {"name": "cat_id", "cardinality": len(sales_train_validation["cat"].unique())},
        {
            "name": "dept_id",
            "cardinality": len(sales_train_validation["dept"].unique()),
        },
        {
            "name": "item_id",
            "cardinality": len(sales_train_validation["item"].unique()),
        },
    ]

    feat_dynamic_real = [
        {"name": "sell_price", "cardinality": 1},
        {"name": "event_1", "cardinality": 1},
        {"name": "event_2", "cardinality": 1},
        {"name": "snap", "cardinality": 1},
    ]

    # Build training set
    train_file = dataset_path / "train" / "data.json"
    train_ds = []
    for index, item in sales_train_validation.iterrows():
        id, item_id, dept_id, cat_id, store_id, state_id = index
        start_index = np.nonzero(item.iloc[:1913].values)[0][0]
        start_date = time_index[start_index]
        time_series = {}

        state_enc, store_enc, cat_enc, dept_enc, item_enc = item.iloc[1913:]

        time_series["start"] = str(start_date)
        time_series["item_id"] = id[:-11]

        time_series["feat_static_cat"] = [
            state_enc,
            store_enc,
            cat_enc,
            dept_enc,
            item_enc,
        ]

        sell_price = get_sell_price(item_id, store_id)
        snap_feature = {
            "CA": snap_CA_feature,
            "TX": snap_TX_feature,
            "WI": snap_WI_feature,
        }[state_id]

        time_series["target"] = (
            item.iloc[start_index:1913].values.astype(np.float32).tolist()
        )
        time_series["feat_dynamic_real"] = (
            np.concatenate(
                (
                    np.expand_dims(sell_price.iloc[start_index:1913].values, 0),
                    event_1_feature[:, start_index:1913],
                    event_2_feature[:, start_index:1913],
                    snap_feature[:, start_index:1913],
                ),
                0,
            )
            .astype(np.float32)
            .tolist()
        )

        train_ds.append(time_series.copy())

    # Build training set
    train_file = dataset_path / "train" / "data.json"
    save_to_file(train_file, train_ds)

    # Create metadata file
    meta_file = dataset_path / "metadata.json"
    with open(meta_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "freq": pandas_freq,
                    "prediction_length": prediction_length,
                    "feat_static_cat": feat_static_cat,
                    "feat_dynamic_real": feat_dynamic_real,
                    "cardinality": len(train_ds),
                }
            )
        )

    # Build testing set
    test_file = dataset_path / "test" / "data.json"
    test_ds = []
    for index, item in sales_train_evaluation.iterrows():
        id, item_id, dept_id, cat_id, store_id, state_id = index
        start_index = np.nonzero(item.iloc[:1941].values)[0][0]
        start_date = time_index[start_index]
        time_series = {}

        state_enc, store_enc, cat_enc, dept_enc, item_enc = item.iloc[1941:]

        time_series["start"] = str(start_date)
        time_series["item_id"] = id[:-11]

        time_series["feat_static_cat"] = [
            state_enc,
            store_enc,
            cat_enc,
            dept_enc,
            item_enc,
        ]

        sell_price = get_sell_price(item_id, store_id)
        snap_feature = {
            "CA": snap_CA_feature,
            "TX": snap_TX_feature,
            "WI": snap_WI_feature,
        }[state_id]

        time_series["target"] = (
            item.iloc[start_index:1941].values.astype(np.float32).tolist()
        )
        time_series["feat_dynamic_real"] = (
            np.concatenate(
                (
                    np.expand_dims(sell_price.iloc[start_index:1941].values, 0),
                    event_1_feature[:, start_index:1941],
                    event_2_feature[:, start_index:1941],
                    snap_feature[:, start_index:1941],
                ),
                0,
            )
            .astype(np.float32)
            .tolist()
        )

        test_ds.append(time_series.copy())

    save_to_file(test_file, test_ds)

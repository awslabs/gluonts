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

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._util import metadata, save_to_file


def generate_walmart_dataset(
    dataset_path: Path, pandas_freq: str, prediction_length: int
):
    def load_dataset(dataset_path):
        train_path = dataset_path / "train.csv"

        if not os.path.exists(train_path):
            raise RuntimeError(
                f"Wallmart data is available on Kaggle"
                f"https://www.kaggle.com/bletchley/course-material-walmart-challenge/download"
                f"Please supply the files at {dataset_path}."
            )

        return pd.read_csv(train_path)

    def create_metadata_file(df, freq, prediction_length, dataset_path):
        # Create metadata file
        cardinalities = [
            len(df[cat].unique()) for cat in ["Store", "Dept", "Type", "Size"]
        ]
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

    def generate_timeseries_data(df):
        ds = []
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Store", "Dept", "Date"])
        for store in df["Store"].unique():
            store_df = df.loc[df["Store"] == store]

            for dept in store_df["Dept"].unique():
                dept_df = store_df.loc[store_df["Dept"] == dept]
                # calc cat_features of timeseries

                cat_features = [
                    int(store),
                    int(dept),
                    dept_df[["Type"]].values[0][0],
                    int(dept_df[["Size"]].values[0][0]),
                ]
                dept_df = dept_df.drop(
                    ["Type", "Size", "Dept", "Store"], axis=1
                )

                # insert rows of NaN for missing dates
                start_date = dept_df.iloc[0]["Date"]
                end_date = dept_df.iloc[-1]["Date"]

                # calc all days between start and end
                dates = pd.date_range(start=start_date, end=end_date, freq="D")
                times = pd.DataFrame({"Date": pd.to_datetime(dates)})
                dept_df = (
                    times.merge(dept_df, on="Date", how="left")
                    .drop(["Date"], axis=1)
                    .T
                )

                # calc target
                target = dept_df.loc["Weekly_Sales"].to_numpy()

                # calc dynamic variables
                dyn_features = dept_df.drop(["Weekly_Sales"]).to_numpy()

                ts = (
                    target,
                    f"{start_date} 00:00:00",
                    dyn_features,
                    cat_features,
                )

                ds.append(ts)
        return ds

    def create_dataset_file(data, cut=None):
        ds_name = "train" if cut else "test"
        to_slice = slice(0, -cut) if cut else slice(None, None, None)

        ds = (
            {
                FieldName.TARGET: target[to_slice].tolist(),
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: fdr.tolist(),
                FieldName.FEAT_STATIC_CAT: fsc,
            }
            for (target, start, fdr, fsc) in data
        )
        path = dataset_path / ds_name / "data.json"
        save_to_file(path, ds)

    df = load_dataset(dataset_path)
    create_metadata_file(df, pandas_freq, prediction_length, dataset_path)

    timeseries_data = generate_timeseries_data(df)
    create_dataset_file(timeseries_data)
    create_dataset_file(timeseries_data, cut=prediction_length)

# For debugging
if __name__ == "__main__":
    pandas_freq = "D"
    user = ""
    dataset_path = Path("/Users/{user}/.mxnet/gluon-ts/datasets/walmart")
    prediction_length = 7
    generate_walmart_dataset(dataset_path, pandas_freq, prediction_length)

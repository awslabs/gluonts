from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets


def generate_ercot_dataset(dataset_path: Path, dataset_writer: DatasetWriter):
    url = "https://github.com/ourownstory/neuralprophet-data/raw/main/datasets_raw/energy/ERCOT_load_2004_2021Sept.csv"
    df = pd.read_csv(url)
    # There is only a single missing value per time series - forward fill them
    df.ffill(inplace=True)
    regions = [col for col in df.columns if col not in ["ds", "y"]]

    freq = "1H"
    prediction_length = 24

    start = pd.Period(df["ds"][0], freq=freq)

    test = [
        {
            "start": start,
            "target": df[region].to_numpy(dtype=np.float64),
        }
        for region in regions
    ]

    train = [
        {
            "start": start,
            "target": df[region].to_numpy(dtype=np.float64)[
                :-prediction_length
            ],
        }
        for region in regions
    ]

    metadata = MetaData(freq=freq, prediction_length=prediction_length)
    dataset = TrainDatasets(metadata=metadata, train=train, test=test)
    dataset.save(str(dataset_path), writer=dataset_writer, overwrite=True)

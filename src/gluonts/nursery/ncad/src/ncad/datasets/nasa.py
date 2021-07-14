# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

from typing import List, Union
from pathlib import PosixPath

import numpy as np
import pandas as pd

import ast

from ncad.ts import TimeSeries, TimeSeriesDataset

from tqdm import tqdm

# path = Path.home()/'ncad_datasets/nasa'

def nasa(
        path: Union[PosixPath,str],
        benchmark: Union[str,List[str]] = ['SMAP','MSL'][0],
        *args, **kwargs ) -> TimeSeriesDataset:
    """ Loads NASA's SMAP and MSL datasets.

    Args:
        path : Path to the directory containing the csv files.
    
    Data source:
        https://github.com/khundman/telemanom
    """
    
    path = PosixPath(path).expanduser()
    assert path.is_dir()
    
    if isinstance(benchmark,str):
        benchmark = [benchmark]

    # Verify that all subdirectories exist
    bmk_dirs = ['train','test']
    assert np.all([(path/bmk_dir).is_dir() for bmk_dir in bmk_dirs])

    # Load labeled anomalies
    # NOTE: These are for the test dataset, according to the github README
    labeled_anomalies_pd = pd.read_csv(path/'labeled_anomalies.csv')

    # Files to be read
    train_files = [fn for fn in os.listdir(path/'train') if fn.endswith('.npy')]
    train_files.sort()
    test_files = [fn for fn in os.listdir(path/'test') if fn.endswith('.npy')]
    test_files.sort()
    assert train_files == test_files

    # Subset labeled_anomalies dataframe
    bmk_idx = labeled_anomalies_pd['spacecraft'].isin( benchmark )
    labeled_anomalies_pd = labeled_anomalies_pd[bmk_idx]

    train_dataset = TimeSeriesDataset()
    test_dataset = TimeSeriesDataset()
    
    for ts_id in tqdm(labeled_anomalies_pd['chan_id']):
        
        fn_i = ts_id+'.npy'
        assert fn_i in train_files
        assert fn_i in test_files

        # Load the multivariate time series from npy files
        ts_train_np = np.load(path/'train'/fn_i)
        ts_test_np = np.load(path/'test'/fn_i)

        T_train, ts_channels_train = ts_train_np.shape
        T_test, ts_channels_test = ts_test_np.shape
        
        assert ts_channels_train == ts_channels_test

        train_anomalies = np.zeros(T_train)
        test_anomalies = np.zeros(T_test)

        # Fetch anomalies from data frame
        anom_row = labeled_anomalies_pd['chan_id'] == ts_id
        if np.any(anom_row):
            anom_seq = labeled_anomalies_pd[anom_row]['anomaly_sequences'].iloc[0]
            anom_seq = ast.literal_eval(str(anom_seq))
            assert isinstance(anom_seq,list)
        for start, end in anom_seq:
            test_anomalies[start:end] = 1

        train_dataset.append(
            TimeSeries(
                values = ts_train_np,
                labels = train_anomalies,
                item_id = f"{ts_id}_train",
            )
        )
        test_dataset.append(
            TimeSeries(
                values = ts_test_np,
                labels = test_anomalies,
                item_id = f"{ts_id}_test",
            )
        )
    
    assert len(train_dataset)>0
    assert len(test_dataset)>0

    return train_dataset, test_dataset

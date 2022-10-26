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
import gluonts.dataset
import torch
import torch as T
import numpy as np
import sklearn.model_selection
import pickle
import os
import torchvision.transforms as transforms
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.split import OffsetSplitter
from utils import calculate_padding_base_2
from gluonts.dataset.rolling_dataset import StepStrategy, NumSplitsStrategy
from gluonts.dataset.rolling_dataset import generate_rolling_dataset


def set_up_data(H):
    # TODO split into train/val/test

    if H.dataset == "dummy":
        n_samples_train_val = 1000
        n_samples_val = 200
        n_samples_test = 500
        # -
        H.n_meas = 5
        n_steps = 10000

        # Version 1): random
        X_train = (
            torch.empty(n_samples_train_val, H.n_meas, n_steps)
            .normal_()
            .float()
            .numpy()
        )  # Conv1D expects (N,C_in,L_in), where C_in=number of channels=measurement dimension, L_in=length of signal sequence
        Y_train = torch.empty(n_samples_train_val).bernoulli_().float().numpy()
        X_test = (
            torch.empty(n_samples_test, H.n_meas, n_steps)
            .normal_()
            .float()
            .numpy()
        )  # Conv1D expects (N,C_in,L_in), where C_in=number of channels=measurement dimension, L_in=length of signal sequence
        Y_test = torch.empty(n_samples_test).bernoulli_().float().numpy()
        # Version 2): all ones
        # X_train = torch.ones(n_samples_train_val, H.n_meas, n_steps).float().numpy()  # Conv1D expects (N,C_in,L_in), where C_in=number of channels=measurement dimension, L_in=length of signal sequence
        # Y_train = torch.ones(n_samples_train_val).float().numpy()
        # X_test = torch.ones(n_samples_test, H.n_meas, n_steps).float().numpy()  # Conv1D expects (N,C_in,L_in), where C_in=number of channels=measurement dimension, L_in=length of signal sequence
        # Y_test = torch.ones(n_samples_test).float().numpy()

    elif H.dataset in list(
        dataset_recipes.keys()
    ):  # all available datasets in gluonts: ['constant', 'exchange_rate', 'solar-energy', 'electricity', 'traffic', 'exchange_rate_nips', 'electricity_nips', 'traffic_nips', 'solar_nips', 'wiki-rolling_nips', 'taxi_30min', 'kaggle_web_traffic_with_missing', 'kaggle_web_traffic_without_missing', 'kaggle_web_traffic_weekly', 'm1_yearly', 'm1_quarterly', 'm1_monthly', 'nn5_daily_with_missing', 'nn5_daily_without_missing', 'nn5_weekly', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly', 'cif_2016', 'london_smart_meters_without_missing', 'wind_farms_without_missing', 'car_parts_without_missing', 'dominick', 'fred_md', 'pedestrian_counts', 'hospital', 'covid_deaths', 'kdd_cup_2018_without_missing', 'weather', 'm3_monthly', 'm3_quarterly', 'm3_yearly', 'm3_other', 'm4_hourly', 'm4_daily', 'm4_weekly', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5', 'uber_tlc_daily', 'uber_tlc_hourly', 'airpassengers']
        # structure of dataset:
        # - contains .train, .test, and .metadata attribute (details see https://ts.gluon.ai/dev/tutorials/forecasting/quick_start_tutorial.html)
        # - .train and .test are iterators, where each entry is a dictionary which contains a few keys:
        #       - 'target' is the time series raw data
        dataset_original = get_dataset(H.dataset)

        dataset_test = dataset_original.test

        # TODO for now:
        # dataset_train, dataset_val = dataset_original.train, dataset_original.train

        # # TODO understand in detail, potentially make s.t. validation data and train data not overlapping
        train_val_splitter = OffsetSplitter(
            prediction_length=(
                (H.forecast_length + H.context_length) * H.val_windows_per_item
            ),
            split_offset=-(
                (H.forecast_length + H.context_length) * H.val_windows_per_item
            ),
            max_history=(H.forecast_length + H.context_length)
            * H.val_windows_per_item,
        )
        if H.single_item:
            data_train = [
                list(dataset_original.train)[H.chosen_id]
            ]  # Alex uses 118 item ID here
            dataset_test = [list(dataset_test)[H.chosen_id]]
        else:
            data_train = dataset_original.train

        train_split, val_split = train_val_splitter.split(data_train)

        # [1] required because first value is a string, second value is what we want
        dataset_train, dataset_val = train_split[1], val_split[1]

        dataset_val = generate_rolling_dataset(
            dataset=dataset_val,
            strategy=NumSplitsStrategy(
                prediction_length=H.forecast_length + H.context_length,
                num_splits=H.val_windows_per_item,
            ),
            start_time=dataset_val[0]["start"],
        )  # only works for electricity, because every item has the same start time

        # print("hello")

        # TODO CORRECT?????????????????
        # dataset_train, dataset_val = gluonts.dataset.DatasetCollection(datasets=dataset_train), gluonts.dataset.DatasetCollection(datasets=dataset_val)

        # doing a rolling prediction dataset for validation data
        # TODO similarly do for test data
        # TODO doesn't work yet

        # entries = list(dataset_original.train)
        # X_list = [entry['target'] for entry in entries]  # list of time series, each of differnt length (hence, cannot convert to numpy array)
        # # insert "measurements channel if not available
        # if X_list[0].ndim == 1:   # only one measurement, otherwise expected that already in (measurements/channel, steps) format
        #     X_list = [np.expand_dims(x, axis=0) for x in X_list]  # new format: (measurements/channel, steps)
        # H.n_meas = X_list[0].shape[0]
        #
        # train_frac, val_frac, test_frac = .7, .15, .15
        # assert train_frac+val_frac+test_frac == 1.
        # # train-val-test split s.t. the sub-datasets are non-overlapping in terms of the original time series
        # X_train = X_list[:int(train_frac*len(X_list))]
        # X_val = X_list[int(train_frac*len(X_list)):int((train_frac+val_frac)*len(X_list))]
        # X_test = X_list[int((train_frac+val_frac)*len(X_list)):int((train_frac+val_frac+test_frac)*len(X_list))]

    elif H.dataset == "sine":

        # OLD -----
        # class SineWaveGluonTSDataset(gluonts.dataset.Dataset):
        #     def __init__(self):
        #         self.n_items = 5000
        #         x = np.linspace(0, 20 * np.pi, 1000)
        #         sin_x = np.sin(x)
        #         # add noise
        #         sin_x = sin_x + np.random.normal(loc=0., scale=0.3, size=sin_x.shape)
        #         # convert to float32, s.t. later compatible with torch model which expects 32 precision
        #         sin_x = sin_x.astype(np.float32)
        #
        #         self.data = []
        #         # all items are ientical
        #         for t in range(self.n_items):
        #             item = {}
        #             item['start'] = 0.  # dummy data
        #             item['target'] = sin_x
        #             item['item_id'] = str(t)
        #             self.data.append(item)
        #
        #     def __iter__(self):
        #         for t in range(self.n_items):
        #             # print(t)
        #             yield self.data[t]
        #         # raise StopIteration
        #
        #     def __len__(self):
        #         return self.n_items
        # OLD -----

        # NEW 1: different noise, different starting point -----
        class SineWaveGluonTSDataset(gluonts.dataset.Dataset):
            def __init__(self):
                self.n_items = 5000

                self.data = []
                # all items are ientical
                for t in range(self.n_items):
                    starting_point = np.random.uniform(0, 1) * 2 * np.pi
                    x = np.linspace(
                        starting_point + 0, starting_point + 20 * np.pi, 1000
                    )
                    sin_x = np.sin(x)
                    # add noise
                    sin_x = sin_x + np.random.normal(
                        loc=0.0, scale=0.3, size=sin_x.shape
                    )
                    # convert to float32, s.t. later compatible with torch model which expects 32 precision
                    sin_x = sin_x.astype(np.float32)

                    item = {}
                    item["start"] = 0.0  # dummy data
                    item["target"] = sin_x
                    item["item_id"] = str(t)
                    self.data.append(item)

            def __iter__(self):
                for t in range(self.n_items):
                    # print(t)
                    yield self.data[t]
                # raise StopIteration

            def __len__(self):
                return self.n_items

        # NEW ----

        # NEW 2: same noise, different starting point -----
        # class SineWaveGluonTSDataset(gluonts.dataset.Dataset):
        #     def __init__(self):
        #         self.n_items = 5000
        #
        #         self.data = []
        #
        #         # noise calculation (first three lines just for the right shape)
        #         starting_point = np.random.uniform(0, 1) * 2 * np.pi
        #         x = np.linspace(starting_point + 0, starting_point + 20 * np.pi, 1000)
        #         sin_x = np.sin(x)
        #         noise = np.random.normal(loc=0., scale=0.3, size=sin_x.shape)
        #
        #         # all items are identical
        #         for t in range(self.n_items):
        #             starting_point = np.random.uniform(0, 1) * 2 * np.pi
        #             x = np.linspace(starting_point + 0, starting_point + 20 * np.pi, 1000)
        #             sin_x = np.sin(x)
        #             # add noise
        #             sin_x = sin_x + noise
        #             # convert to float32, s.t. later compatible with torch model which expects 32 precision
        #             sin_x = sin_x.astype(np.float32)
        #
        #             item = {}
        #             item['start'] = 0.  # dummy data
        #             item['target'] = sin_x
        #             item['item_id'] = str(t)
        #             self.data.append(item)
        #
        #
        #     def __iter__(self):
        #         for t in range(self.n_items):
        #             # print(t)
        #             yield self.data[t]
        #         # raise StopIteration
        #
        #     def __len__(self):
        #         return self.n_items
        # NEW ----

        dataset_train = SineWaveGluonTSDataset()
        dataset_val = SineWaveGluonTSDataset()
        dataset_test = SineWaveGluonTSDataset()

    elif H.dataset == "2gauss":

        class TwoGaussians(gluonts.dataset.Dataset):
            def __init__(self):
                self.n_items = 50  # must be divisible by 2

                self.data = []

                # all items are identical; this order of for loops for shuffeling reasons
                for t in range(int(self.n_items / 2)):
                    for i in range(2):
                        if i == 0:
                            mean = -1.0
                        else:
                            mean = 1.0
                        noise = np.random.normal(
                            loc=mean, scale=0.3, size=(1000)
                        )
                        noise = noise.astype(np.float32)

                        item = {
                            'start': 0.0,  # dummy data
                            'target': noise,
                            'item_id': str(t)
                        }
                        self.data.append(item)

            def __iter__(self):
                for t in range(self.n_items):
                    # print(t)
                    yield self.data[t]
                # raise StopIteration

            def __len__(self):
                return self.n_items

            # NEW ----

        dataset_train = TwoGaussians()
        dataset_val = TwoGaussians()
        dataset_test = TwoGaussians()

    # only used for the dummy datase
    if H.dataset == "dummy":
        try:
            X_val
        except NameError:
            # split train data into train and test
            # currently for all datasets
            X_train, X_val = sklearn.model_selection.train_test_split(
                X_train, test_size=n_samples_val, random_state=H.seed
            )

    # for normalizing the data, computed per measurements (over all time series and steps)
    if H.dataset in list(dataset_recipes.keys()) or H.dataset in [
        "sine",
        "2gauss",
    ]:
        if H.dataset in list(dataset_recipes.keys()):
            entries = list(dataset_train)
        elif H.dataset in ["sine", "2gauss"]:
            entries = [
                dataset_train.data[t] for t in range(dataset_train.n_items)
            ]
        X_list = [
            entry["target"] for entry in entries
        ]  # list of time series, each of differnt length (hence, cannot convert to numpy array)
        item_id_list = [entry["item_id"] for entry in entries]
        X_train = np.vstack(X_list)

        # insert "measurements channel if not available
        if (
            X_train.ndim == 2
        ):  # only one measurement, otherwise expected that already in (measurements/channel, steps) format
            X_train = np.expand_dims(
                X_train, axis=1
            )  # new format: (measurements/channel, steps)
            X_list = [np.expand_dims(x, axis=0) for x in X_list]

        # define H.meas here!
        H.n_meas = X_train.shape[1]

    if H.normalize == "per_ts_standardize":
        id_to_norm_mean, id_to_norm_std = {}, {}
        for (x, id) in zip(X_list, item_id_list):
            norm_mean = torch.from_numpy(
                (-np.mean(x, axis=1)).reshape((-1, 1))
            )
            norm_std = torch.from_numpy(
                (1.0 / np.std(x, axis=1)).reshape((-1, 1))
            )
            if "cuda" in H.device:
                norm_mean = norm_mean.cuda()
                norm_std = norm_std.cuda()
            id_to_norm_mean[id] = norm_mean
            id_to_norm_std[id] = norm_std
    elif H.normalize == "train_data_standardize":
        norm_mean = torch.from_numpy(
            (-np.mean(X_train, axis=(0, 2))).reshape((1, -1, 1))
        )
        norm_std = torch.from_numpy(
            (1.0 / np.std(X_train, axis=(0, 2))).reshape((1, -1, 1))
        )
        if "cuda" in H.device:
            norm_mean = norm_mean.cuda()
            norm_std = norm_std.cuda()

    def get_normalize_mean_std(ids):
        nonlocal norm_mean, norm_std, id_to_norm_mean, id_to_norm_std
        if H.normalize == "per_ts_standardize":
            norm_neg_mean_list, norm_std_list = [], []
            for id in ids:
                norm_neg_mean_list.append(id_to_norm_mean[id])
                norm_std_list.append(id_to_norm_std[id])
            mean = torch.stack(norm_neg_mean_list)
            std = torch.stack(norm_std_list)
        elif H.normalize == "train_data_standardize":
            mean = norm_mean
            std = norm_std

        return mean, std

    def normalize_fn(x, ids):
        """
        Given an item from the TensorDataset, this function creates two versions of it: one used as input of the model, one used as the target of the model prediction.
        """
        # compute mean and std normalization statistics
        mean, std = get_normalize_mean_std(ids)
        # print("normalize", mean[0], std[0])
        # do normalization
        x.add_(mean).mul_(std)

        # add padding, if necessary. pads to the next power of two in the first dimension
        pad = (
            H.pad_forecast
            if x.shape[2] == H.forecast_length
            else H.pad_context
        )  # if H.forecast_length == H.context_length: both will be padded with the same anyway
        if pad > 0:
            x = torch.nn.functional.pad(
                x, pad=(pad, 0)
            )  # only pad left, only pad last dimension (time dimension); (by default) zero padding

        # Note: x_input has channels as dim 1, x_target has channels as dim 3  # TODO ??? is this true? what does this mean?
        return x

    def unnormalize_fn(x, ids):
        # compute mean and std normalization statistics
        mean, std = get_normalize_mean_std(ids)
        # print("unnormalize", mean[0], std[0])
        # invert mean and std, since parameters for normalization
        mean, std = -mean, 1.0 / std
        # do unnormalization
        x.mul_(std).add_(mean)  # mind the order

        # remove padding
        # TODO temporarily just done for fc VDVAE
        # TODO temporarily taken out since un-padding done in likelihood model
        # if H.model == 'vdvae_fc':
        #     pad = H.pad_forecast if x.shape[2] == H.forecast_length else H.pad_context
        #     x = x[:, :, pad:]  # only un-pad left, only un-pad last dimension (time dimension)

        return x

    # compute how much to pad in preprocessing
    pad_context, pad_forecast = calculate_padding_base_2(H)
    H.pad_context = pad_context
    H.pad_forecast = pad_forecast

    if H.dataset == "dummy":
        dataset_train = TimeSeriesDataset(H, X_train)
        dataset_val = TimeSeriesDataset(H, X_val)
        dataset_test = TimeSeriesDataset(H, X_test)

    return (
        dataset_train,
        dataset_val,
        dataset_test,
        normalize_fn,
        unnormalize_fn,
    )


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Possible extensions:
    - downsampling (only taking every x-th point in the time series)
    """

    def __init__(self, H, X_list):
        # 'index' (without a further specification) refers to the global index, used by __get_item__(...)
        self.X_list = X_list
        self.ts_indices = list(range(len(X_list)))
        self.index_to_ts_index = {}
        self.index_to_within_ts_index = (
            {}
        )  # start position of the window corresponding to index
        self.n_indices = None
        self.context_length = H.context_length
        self.forecast_length = H.forecast_length
        self.full_window_length = H.forecast_length + H.context_length
        self.conditional = H.conditional

        # populate index_to_ts_index and index_to_within_ts_index
        index_count = 0
        for s, x in enumerate(X_list):
            ts_length = x.shape[1]  # x has shape (meas, steps)
            new_indices = ts_length - self.full_window_length
            for i in range(index_count, index_count + new_indices):
                self.index_to_ts_index[i] = s
                self.index_to_within_ts_index[i] = i - index_count

            # increment index count with
            index_count += new_indices

        # populate total number of indices
        self.n_indices = index_count

    def __len__(self):

        return self.n_indices

    def __getitem__(self, index):
        start = self.index_to_within_ts_index[index]

        # Note: Could do in separate classes to avoid the branching.
        if self.conditional:
            x_context = self.X_list[self.index_to_ts_index[index]][
                :, start : start + self.context_length
            ]
            x_forecast = self.X_list[self.index_to_ts_index[index]][
                :,
                start
                + self.context_length : start
                + self.context_length
                + self.forecast_length,
            ]
            # TODO adapt main to reflect two inputs
            # TODO adapt so that can nicely switch between two outputs vs. one output

            return x_context, x_forecast
        else:
            x = self.X_list[self.index_to_ts_index[index]][
                :, start : start + self.full_window_length
            ]

            return x

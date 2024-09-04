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

from typing import Optional

from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict

from ncad.ts import TimeSeriesDataset, ts_random_crop


class NCADDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ts_dataset: TimeSeriesDataset,
        validation_ts_dataset: Optional[TimeSeriesDataset],
        test_ts_dataset: Optional[TimeSeriesDataset],
        window_length: int,
        suspect_window_length: int,
        num_series_in_train_batch: int,
        num_crops_per_series: int = 1,
        label_reduction_method: Optional[str] = [None, "any"][-1],
        stride_val_test: int = 1,
        num_workers: int = 0,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.train_ts_dataset = train_ts_dataset
        self.validation_ts_dataset = validation_ts_dataset
        self.test_ts_dataset = test_ts_dataset

        hparams = AttributeDict(
            window_length=window_length,
            suspect_window_length=suspect_window_length,
            num_series_in_train_batch=num_series_in_train_batch,
            num_crops_per_series=num_crops_per_series,
            label_reduction_method=label_reduction_method,
            stride_val_test=stride_val_test,
            num_workers=num_workers,
        )
        self.hparams = hparams

        self.datasets = {}
        assert (
            not train_ts_dataset.nan_ts_values
        ), "TimeSeries in train_ts_dataset must not have nan values."
        self.datasets["train"] = CroppedTimeSeriesDatasetTorch(
            ts_dataset=train_ts_dataset,
            window_length=self.hparams.window_length,
            suspect_window_length=self.hparams.suspect_window_length,
            label_reduction_method=self.hparams.label_reduction_method,
            num_crops_per_series=self.hparams.num_crops_per_series,
        )

        if validation_ts_dataset is not None:
            assert (
                not validation_ts_dataset.nan_ts_values
            ), "TimeSeries in validation_ts_dataset must not have nan values."
            self.datasets["validation"] = TimeSeriesDatasetTorch(validation_ts_dataset)

        if test_ts_dataset is not None:
            assert (
                not test_ts_dataset.nan_ts_values
            ), "TimeSeries in test_ts_dataset must not have nan values."
            self.datasets["test"] = TimeSeriesDatasetTorch(test_ts_dataset)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.hparams[f"num_series_in_train_batch"],
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets["validation"],
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class TimeSeriesDatasetTorch(Dataset):
    """Time series dataset

    Creates a pytorch dataset based on a TimeSeriesDataset.

    It is possible to apply transformation to the input TimeSeries or the windows.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
    ) -> None:
        """
        Args:
            dataset : TimeSeriesDataset with which serve as the basis for the Torch dataset.
        """
        self.dataset = dataset

        self.transform = Compose(
            [
                Lambda(lambda ts: [ts.values, ts.labels]),
                Lambda(
                    lambda vl: [np.expand_dims(vl[0], axis=1) if vl[0].ndim == 1 else vl[0], vl[1]]
                ),  # Add ts channel dimension, if needed
                Lambda(
                    lambda vl: [np.transpose(vl[0]), vl[1]]
                ),  # Transpose ts values, so the dimensions are (channel, time)
                Lambda(lambda x: [torch.from_numpy(x_i) for x_i in x]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.transform(self.dataset[idx])

        return x, y


class CroppedTimeSeriesDatasetTorch(Dataset):
    """Cropped time series dataset

    Creates a pytorch dataset based on windows from a TimeSeriesDataset.

    Each window (a.k.a. crop) has length of window_length.

    The label y is based on the last 'suspect_window_length' time steps.
    The labels are aggregated according to label_reduction_method.

    It is possible to apply transformation to the input TimeSeries or each window.
    """

    def __init__(
        self,
        ts_dataset: TimeSeriesDataset,
        window_length: int,
        suspect_window_length: int,
        num_crops_per_series: int = 1,
        label_reduction_method: Optional[str] = [None, "any"][-1],
    ) -> None:
        """
        Args:
            ts_dataset : TimeSeriesDataset with which serve as the basis for the cropped windows
            window_length : Length of the (random) windows to be considered. If not specified, the whole series is returned.
            suspect_window_length : Number of timesteps considered at the end of each window
                to define whether a window is anomalous of not.
            num_crops_per_series : Number of random windows taken from each TimeSeries from dataset.
            label_reduction_method : Method used to reduce the labels in the suspect window.
                None : All labels in the suspect window are returned
                'any' : The anomalies of a window is anomalous is any timestep in the suspect_window_length is marked as anomalous.
        """
        self.ts_dataset = ts_dataset

        self.window_length = int(window_length) if window_length else None

        self.suspect_window_length = int(suspect_window_length)
        self.label_reduction_method = label_reduction_method

        self.num_crops_per_series = int(num_crops_per_series)

        # Validate that all TimeSeries in ts_dataset are longer than window_length
        ts_dataset_lengths = np.array([len(ts.values) for ts in self.ts_dataset])
        if any(ts_dataset_lengths < self.window_length):
            raise ValueError(
                "All TimeSeries in 'ts_dataset' must be of length greater or equal to 'window_length'"
            )

        self.cropping_fun = partial(
            ts_random_crop, length=self.window_length, num_crops=self.num_crops_per_series
        )

        # Apply ts_window_transform, to anomalize the window randomly
        self.transform = Compose(
            [
                # Pick a random crop from the selected TimeSeries
                Lambda(lambda x: self.cropping_fun(ts=x)),  # Output: List with cropped TimeSeries
                Lambda(
                    lambda x: (
                        np.stack([ts.values.reshape(ts.shape).T for ts in x], axis=0),
                        np.stack([ts.labels for ts in x], axis=0),
                    )
                ),  # output: tuple of two np.arrays (values, labels), with shapes (num_crops, N, T) and (num_crops, T)
                Lambda(
                    lambda x: [torch.from_numpy(x_i) for x_i in x]
                ),  # output: two torch Tensor (values, labels) with shapes (num_crops, N, T) and (num_crops, T)
            ]
        )

    def __len__(self):
        return len(self.ts_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.transform(self.ts_dataset[idx])

        y_suspect = reduce_labels(
            y=y,
            suspect_window_length=self.suspect_window_length,
            reduction_method=self.label_reduction_method,
        )

        return x, y_suspect


def reduce_labels(
    y: torch.Tensor,
    suspect_window_length: int,
    reduction_method: Optional[str] = [None, "any"][-1],
) -> torch.Tensor:
    """Auxiliary function to reduce labels, one per batch element

    Args:
        y : Tensor with the labels to be reduced. Shape (batch, time).
        suspect_window_length : Number of timesteps considered at the end of each window
            to define whether a window is anomalous of not.
        reduction_method : Method used to reduce the labels in the suspect window.
            None : All labels in the suspect window are returned. The output is a 2D tensor.
            'any' : The anomalies of a window is anomalous if any timestep in the
                    suspect_window_length is marked as anomalous. The output is a 1D tensor.
    Output:
        y_suspect : Tensor with the reduced labels. Shape depends on the reduction_method used.
    """

    suspect_window_length = int(suspect_window_length)

    y_suspect = y[..., -suspect_window_length:]

    if reduction_method is None:
        pass
    elif reduction_method == "any":
        # Currently we will do:
        #   nan are valued as zero, unless
        #   if there are only nan's, y will be nan
        #     [0,0,0,0,0] -> 0
        #     [0,0,0,1,0] -> 1
        #     [nan,nan,nan,nan,nan] -> nan
        #     [0,0,0,nan,0] -> nan
        #     [0,nan,0,1,0] -> 1
        #     [nan,nan,nan,1,nan] -> 1
        y_nan = torch.isnan(y_suspect)
        if torch.any(y_nan).item():
            y_suspect = torch.where(
                y_nan, torch.zeros_like(y_suspect), y_suspect
            )  # Substitutes nan by 0
            y_suspect = (
                torch.sum(y_suspect, dim=1).bool().float()
            )  # Add to check if theres any 1 in each row
            y_suspect = torch.where(
                torch.sum(y_nan, dim=1).bool(), torch.full_like(y_suspect, float("nan")), y_suspect
            )  # put nan if all elements are nan
        else:
            y_suspect = torch.sum(y_suspect, dim=1).bool().float()
    else:
        raise ValueError(f"reduction_method = {reduction_method} not supported.")

    return y_suspect

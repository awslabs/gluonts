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
import logging
from random import choices, randint

import torch
from torch.utils.data import Dataset

import gluonts

# from gluonts.core.component import validated
from gluonts.transform import Chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class SubSeriesDatasetCNN(Dataset):
    # @validated() #TODO: Fix the validated runtime error
    def __init__(
        self,
        dataset: gluonts.dataset.common.Dataset,
        transformation: Chain,
        nb_negative_samples: int,
        device: torch.device,
        max_len: int = None,
        batch_size: int = 1,
        scaling: str = "global",
    ):
        assert (
            scaling == "local" or scaling == "global"
        ), "scaling has to be \
            local or global. If it is local, then the whole time series will be\
            mix-max scaled. If it is local, then each subseries of length \
            target_len will be min-max scaled independenty."
        self.scaling = scaling
        self.device = device
        self.dataset_list_cpu = list(transformation(dataset, is_train=False))
        self.dataset_list = {}
        for idx, elmnt in enumerate(self.dataset_list_cpu):
            x = torch.tensor(elmnt["target"], device=device)
            self.dataset_list[str(idx)] = {
                "target": x
                if scaling == "local"
                else self._min_max_scaling(x.unsqueeze(0)).squeeze(0),
                "time_feat": torch.tensor(elmnt["time_feat"], device=device),
                "feat_static_cat": torch.tensor(
                    elmnt["feat_static_cat"], device=device
                ),
                "observed_values": torch.tensor(
                    elmnt["observed_values"], device=device
                ),
            }
        self.nb_negative_samples = nb_negative_samples
        self.max_len = max_len
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset_list)

    def _min_max_scaling(self, target):
        _min, _ = torch.min(target, dim=1)
        _max, _ = torch.max(target, dim=1)

        _min = _min.unsqueeze(dim=1)

        _max = _max.unsqueeze(dim=1)
        scaling_factor = _max - _min
        scaling_factor = torch.where(
            scaling_factor != 0,
            scaling_factor,
            torch.ones_like(scaling_factor, device=self.device),
        )
        scaled_target = (target - _min) / scaling_factor
        return scaled_target

    def _pick_sublist(self, tensor: torch.Tensor, _start: int, _length: int):
        """Pick subtensor of a torch.Tensor

        Arguments:
            tensor:
                torch.Tensor objetc of shape (1, L)

        Returns:
            A torch.Tensor object of shape (1, _length) extracted from tensor,
            starting at the index _start

        Example:
            Input:
                torch.tensor([[0,1,2,3,4,5,]]), 1, 2
            Output:
                torch.tensor([[1,2,3]])
        """
        assert _start + _length <= tensor.size(
            1
        ), "start + length is strictly greater than the length of the list"
        assert _start >= 0, "starting index is negative"
        assert _length > 0, "length is null or negative"

        return tensor[:, _start : _start + _length]

    def _pick_negative_subseries(self, index, _maxi_len):
        idx_neg = choices(
            [i for i in range(len(self)) if i != index],
            k=self.nb_negative_samples,
        )  # Pick time series indexes from the dataset, excluding the current one from which we sampled the context
        negative_time_series = [
            self.dataset_list[str(idx_neg[k])]["target"].view(1, -1)
            for k in range(self.nb_negative_samples)
        ]  # Pick the time series
        len_negative_ts = [ts.size(1) for ts in negative_time_series]
        min_len = min(len_negative_ts)

        if (
            index % self.batch_size == 0
        ):  # At every new batch, sample new size for the negative sub-series
            self.size_neg = self._set_size(1, min(min_len + 1, _maxi_len))
        size_neg = self.size_neg

        neg_samples = torch.empty(
            self.nb_negative_samples, size_neg, device=self.device
        )

        if size_neg == min_len:
            # start_neg = torch.zeros(self.nb_negative_samples, dtype=int, device = self.device)
            start_neg = [0 for k in range(self.nb_negative_samples)]
        else:
            start_neg = choices(
                [i for i in range(min_len - size_neg)],
                k=self.nb_negative_samples,
            )

        for k in range(self.nb_negative_samples):
            neg_samples[k, :] = self._pick_sublist(
                negative_time_series[k], start_neg[k], size_neg
            ).squeeze()

        return neg_samples

    def _set_size(self, start, end):
        size = randint(start, end - 1)
        return size

    def __getitem__(self, index: int):
        if self.max_len is None:
            _maxi_len = float("inf")
        else:
            _maxi_len = self.max_len

        time_series = self.dataset_list[str(index)]["target"].view(1, -1)

        if (
            index % self.batch_size == 0
        ):  # At every batch, sample a new size for the positive sub-series and a new size for the context.

            self.size_positive = self._set_size(
                1, min(time_series.size(1) + 1, _maxi_len)
            )
            self.size_context = self._set_size(
                self.size_positive, min(time_series.size(1) + 1, _maxi_len)
            )
        size_context = self.size_context
        size_positive = self.size_positive

        start_context = randint(0, time_series.size(1) - size_context - 1)
        context = self._pick_sublist(time_series, start_context, size_context)

        if size_positive == size_context:
            start_positive = 0
        else:
            start_positive = randint(0, size_context - size_positive - 1)
        positive = self._pick_sublist(context, start_positive, size_positive)

        neg_samples = self._pick_negative_subseries(index, _maxi_len)

        if self.scaling == "local":
            context = self._min_max_scaling(context)
            positive = self._min_max_scaling(positive)
            neg_samples = self._min_max_scaling(neg_samples)

        return (context, positive, neg_samples)

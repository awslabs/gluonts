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
from itertools import groupby
from random import choice, randint
from time import time

import torch
from torch.utils.data import Dataset

# from gluonts.core.component import validated
from gluonts.dataset.common import Dataset as gluonDS
from gluonts.transform import Chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class Data(Dataset):
    # @validated()
    def __init__(
        self,
        dataset: gluonDS,
        transformation: Chain,
        nb_samples: int,
        target_len: int,
        batch_size: int,
        device: torch.device,
        scaling: str = "local",
        context_length: int = 0,
        exclude_index: list = None,
    ):
        super(Data, self).__init__()
        assert (
            scaling == "local" or scaling == "global" or scaling == "NoScale"
        ), "scaling has to be \
            local or global. If it is local, then the whole time series will be\
            mix-max scaled. If it is local, then each subseries of length \
            target_len will be min-max scaled independenty. If is is NoScale\
            then no scaling is applied to the dataset."
        self.dataset_list_cpu = list(transformation(dataset, is_train=False))
        self.len_data_list = len(self.dataset_list_cpu)
        self.dataset_list = {}
        self.nb_samples = nb_samples
        self.target_len = target_len
        self.batch_size = batch_size
        self.device = device
        self.context_length = context_length
        for idx, elmnt in enumerate(self.dataset_list_cpu):
            x = torch.tensor(elmnt["target"], device=device)
            self.dataset_list[str(idx)] = {
                "target": x
                if scaling == "local" or scaling == "NoScale"
                else self._min_max_scaling(x.unsqueeze(0)).squeeze(0),
                "time_feat": torch.tensor(elmnt["time_feat"], device=device),
                "feat_static_cat": torch.tensor(
                    elmnt["feat_static_cat"], device=device
                ),
                "observed_values": torch.tensor(
                    elmnt["observed_values"], device=device
                ),
                "sliced_array": self._slice_array(
                    elmnt["observed_values"], device
                ),
            }

        self.index_to_sample_from = [
            str(i)
            for i in range(0, self.len_data_list)
            if i not in exclude_index
        ]
        self.zero = torch.zeros(1, device=device)

    def _slice_array(self, arr, device):
        """
        Takes as input an array like [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0]
        Returns a Tuple:
            - [{'start': torch.tensor(0), 'end': torch.tensor(3)},
                {'start': torch.tensor(8), 'end': torch.tensor(11)},
            ]
            - [torch.tensor(4), torch.tensor(4)]
        """

        s = "".join(str(int(k)) for k in arr)
        L = [[int(i) for i in list(g)] for k, g in groupby(s)]
        TOTAL_L = 0
        L2 = []
        L_length = []
        for i, l in enumerate(L):
            if 1 in l:
                s = TOTAL_L
                e = TOTAL_L + len(l) - 1
                if (
                    torch.tensor(len(l), device=device)
                    > self.target_len + self.context_length
                ):
                    L2.append(
                        {
                            "start": torch.tensor(s, device=device),
                            "end": torch.tensor(e, device=device),
                        }
                    )
                    L_length.append(
                        (
                            torch.tensor(len(L_length), device=device),
                            torch.tensor(len(l), device=device),
                        )
                    )
            TOTAL_L += len(l)
        return L2, L_length

    def _min_max_scaling(self, target):
        _min, _ = torch.min(target, dim=1)
        _max, _ = torch.max(target, dim=1)

        _min = _min.unsqueeze(dim=1)

        _max = _max.unsqueeze(dim=1)
        scaling_factor = _max - _min
        scaling_factor = torch.where(
            scaling_factor != 0,
            scaling_factor,
            torch.ones_like(scaling_factor),
        )
        scaled_target = (target - _min) / scaling_factor
        return scaled_target

    def __len__(self):
        return self.nb_samples * self.batch_size

    def _choose_section(self, index):
        start_end_section, length = self.dataset_list[index]["sliced_array"]
        idx, length_section = choice(length)
        return start_end_section[idx]["start"], start_end_section[idx]["end"]

    def _get_start_end(self, index):
        start_section, end_section = self._choose_section(index)
        start = randint(
            start_section + self.context_length,
            end_section - self.target_len + 1,
        )  # Start index of sub-series
        return start, start + self.target_len

    def _get_noise_with_time_feat(self, time_feat):
        noise = torch.randn((1, self.target_len), device=self.device)
        noise_time_feat = torch.cat((time_feat, noise))
        return noise_time_feat

    def __getitem__(self, idx):

        index = choice(self.index_to_sample_from)

        start, end = self._get_start_end(index)
        if self.context_length > 0:
            context = self.dataset_list[index]["target"][
                start - self.context_length : start
            ]
        else:
            context = torch.empty((1, 1), device=self.device)
        target = self.dataset_list[index]["target"][start:end]
        time_feat = self.dataset_list[index]["time_feat"][:, start:end]
        noise_time_feat = self._get_noise_with_time_feat(time_feat)
        feat_static_cat = self.dataset_list[index]["feat_static_cat"]
        return target, time_feat, noise_time_feat, feat_static_cat, context

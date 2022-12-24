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

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence

from .sampling import Triplet
from .dataset import TimeSeries
from meta.common.torch import tensor_to_np


@dataclass
class SeriesBatch:
    """
    A batch of series from different base datasets, represented by the padded batch, the lengths of the series
    and the splits sizes indicating the corresponding base dataset.
    """

    sequences: torch.Tensor  # shape [batch, num_sequences, max_sequence_length]
    lengths: torch.Tensor  # shape [batch]
    split_sections: torch.Tensor  # shape [batch]
    scales: Optional[
        torch.Tensor
    ] = None  # shape[batch, 2] contains mean and std the ts has been scaled with

    @classmethod
    def from_lists(
        cls, lists: List[List[TimeSeries]], squeeze: bool = False
    ) -> SeriesBatch:
        """
        Initializes a series batch from the provided series.

        Args:
            lists: List of lists of time series.
            squeeze: If the last dimension should be squeezed. This should be done for future queries
                which do not contain co-variates and are, thus, univariate

        Returns:
            The initialized series batch.
        """
        split_sections = [len(section) for section in lists]
        series = [s for sublist in lists for s in sublist]
        if squeeze:
            values = [torch.squeeze(s.values, -1) for s in series]
        else:
            values = [s.values for s in series]
        return SeriesBatch(
            pad_sequence(values, batch_first=True),
            lengths=torch.as_tensor([len(s) for s in series]),
            split_sections=torch.as_tensor(split_sections),
            scales=torch.stack([s.scale for s in series])
            if series[0].scale is not None
            else None,
        )

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.lengths = self.lengths.pin_memory()
        self.split_sections = self.split_sections.pin_memory()
        if self.scales is not None:
            self.scales = self.scales.pin_memory()
        return self

    def unpad(self, to_numpy: bool = False, squeeze: bool = False):
        """
        Unpad sequences in batch by using the lengths.
        """
        t = tensor_to_np if to_numpy else lambda x: x
        sq = torch.squeeze if squeeze else lambda x: x

        splits = torch.split(self.sequences, self.split_sections.tolist())
        lengths = torch.split(self.lengths, self.split_sections.tolist())
        out = []
        for l_split, split in zip(lengths, splits):
            un_padded = [t(sq(s[:l])) for l, s in zip(l_split, split)]
            if squeeze:
                out.extend(un_padded)
            else:
                out.append(un_padded)
        return out

    def to(self, device) -> SeriesBatch:
        return SeriesBatch(
            self.sequences.to(device),
            self.lengths.to(device),
            self.split_sections.to(device),
            scales=self.scales.to(device) if self.scales is not None else None,
        )

    def rescale(self) -> SeriesBatch:
        """
        Redo standardization. The series must contain the same time series in the same order as the dataset.
        """
        m = self.scales[:, 0].unsqueeze(1)
        std = self.scales[:, 1].unsqueeze(1)
        return SeriesBatch(
            self.sequences * std + m,
            self.lengths,
            self.split_sections,
        )

    def one_per_split(self) -> SeriesBatch:
        """
        Choose the first element of every split section and return the resulting series batch.
        This method should only be used on query (past and future) series batches.
        """
        splits = torch.split(self.sequences, self.split_sections.tolist())
        lengths = torch.split(self.lengths, self.split_sections.tolist())
        return SeriesBatch(
            sequences=torch.cat(tuple(split[0:1] for split in splits)),
            lengths=torch.cat(tuple(l[0:1] for l in lengths)),
            split_sections=torch.ones(len(splits), dtype=int),
        )

    def first_n(self, n: int) -> SeriesBatch:
        """
        Choose the first n splits of the sequences as defined by split_sections.
        Self.sequences has thus sum(split_section[i], i=0, ..., n-1) elements.
        """
        splits = torch.split(self.sequences, self.split_sections.tolist())
        lengths = torch.split(self.lengths, self.split_sections.tolist())
        return SeriesBatch(
            sequences=torch.cat(splits[:n]),
            lengths=torch.cat(lengths[:n]),
            split_sections=self.split_sections[:n],
        )

    def __getitem__(self, index: int) -> SeriesBatch:
        split = torch.split(self.sequences, self.split_sections.tolist())[
            index
        ]
        length = torch.split(self.lengths, self.split_sections.tolist())[index]
        return SeriesBatch(
            sequences=split,
            lengths=length,
            split_sections=self.split_sections[index : index + 1],
        )


@dataclass
class TripletBatch:
    """
    A triplet batch, composed of a batch of support sets, query contexts and prediction horizons.
    Compared to a simple triplet, it also manages the lengths of all samples.
    """

    support_set: SeriesBatch
    query_past: SeriesBatch
    query_future: SeriesBatch

    @classmethod
    def collate(cls, triplets: List[Triplet]) -> TripletBatch:
        """
        Combines a list of triplets into a batched triplet to pass to a network.
        """
        s, p, f = zip(*triplets)
        return TripletBatch(
            SeriesBatch.from_lists(list(s)),
            SeriesBatch.from_lists(list(p)),
            SeriesBatch.from_lists(list(f), squeeze=True),
        )

    def to(self, device) -> TripletBatch:
        return TripletBatch(
            self.support_set.to(device),
            self.query_past.to(device),
            self.query_future.to(device),
        )

    def pin_memory(self):
        self.support_set = self.support_set.pin_memory()
        self.query_past = self.query_past.pin_memory()
        self.query_future = self.query_future.pin_memory()
        return self

    def reduce_to_unique_query(self) -> TripletBatch:
        """
        Selects the first query of every group of queries that use the same support set
        (see split_sections of queries) and returns the resulting triplet batch.
        Query past and future are reduced, support sets are not touched.
        """
        return TripletBatch(
            support_set=self.support_set,
            query_past=self.query_past.one_per_split(),
            query_future=self.query_future.one_per_split(),
        )

    def first_n(self, n: int) -> TripletBatch:
        """
        Choose the first n splits of each series batch and return the resulting triplet batch.
        """
        return TripletBatch(
            support_set=self.support_set.first_n(n),
            query_past=self.query_past.first_n(n),
            query_future=self.query_future.first_n(n),
        )

    def __getitem__(self, index: int) -> TripletBatch:
        return TripletBatch(
            support_set=self.support_set[index],
            query_past=self.query_past[index],
            query_future=self.query_future[index],
        )

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

# Standard library imports
from typing import Iterator, List

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.transform import FlatMapTransformation, shift_timestamp


class ForkingSequenceSplitter(FlatMapTransformation):
    """Forking sequence splitter."""

    @validated()
    def __init__(
        self,
        train_sampler,
        enc_len: int,
        dec_len: int,
        time_series_fields: List[str] = None,
        target_in="target",
        is_pad_out: str = "is_pad",
        start_in: str = "start",
        forecast_start_out: str = "forecast_start",
    ) -> None:
        assert enc_len > 0, "The value of `enc_len` should be > 0"
        assert dec_len > 0, "The value of `dec_len` should be > 0"

        self.train_sampler = train_sampler
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.ts_fields = (
            time_series_fields if time_series_fields is not None else []
        )
        self.target_in = target_in
        self.is_pad_out = is_pad_out
        self.start_in = start_in
        self.forecast_start_out = forecast_start_out

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        dec_len = self.dec_len
        slice_cols = self.ts_fields + [self.target_in]
        target = data[self.target_in]

        if is_train:
            if len(target) < self.dec_len:
                # We currently cannot handle time series that are shorter than the
                # prediction length during training, so we just skip these.
                # If we want to include them we would need to pad and to mask
                # the loss.
                sampling_indices: List[int] = []
            else:
                sampling_indices = self.train_sampler(
                    target, 0, len(target) - self.dec_len
                )
        else:
            sampling_indices = [len(target)]

        for i in sampling_indices:
            pad_length = max(self.enc_len - i, 0)

            d = data.copy()
            for ts_field in slice_cols:
                if i > self.enc_len:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.enc_len : i]
                elif i < self.enc_len:
                    pad_block = np.zeros(
                        d[ts_field].shape[:-1] + (pad_length,)
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    past_piece = d[ts_field][..., :i]

                d[self._past(ts_field)] = np.expand_dims(past_piece, -1)

                if is_train and ts_field is self.target_in:
                    forking_dec_field = np.zeros(
                        shape=(self.enc_len, self.dec_len)
                    )

                    for j in range(self.enc_len):
                        start_idx = i - self.enc_len + j + 1
                        if start_idx >= 0:
                            forking_dec_field[j, :] = d[ts_field][
                                ..., start_idx : start_idx + dec_len
                            ]

                    d[self._future(ts_field)] = forking_dec_field

                del d[ts_field]

            pad_indicator = np.zeros(self.enc_len)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1
            d[self._past(self.is_pad_out)] = pad_indicator
            d[self.forecast_start_out] = shift_timestamp(d[self.start_in], i)
            yield d

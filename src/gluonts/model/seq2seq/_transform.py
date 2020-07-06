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
from collections import Counter
from typing import Any, Iterator, List, Optional

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import FlatMapTransformation, shift_timestamp


def pad_to_size(xs, size):
    """Pads `xs` with 0 on the left on the last axis."""
    pad_length = size - xs.shape[-1]
    if pad_length <= 0:
        return xs

    pad_width = ([(0, 0)] * (xs.ndim - 1)) + [(pad_length, 0)]
    return np.pad(xs, mode="constant", pad_width=pad_width)


class ForkingSequenceSplitter(FlatMapTransformation):
    """Forking sequence splitter."""

    @validated()
    def __init__(
        self,
        train_sampler,
        enc_len: int,
        dec_len: int,
        target_field=FieldName.TARGET,
        encoder_series_fields: Optional[List[str]] = None,
        decoder_series_fields: Optional[List[str]] = None,
        prediction_time_decoder_exclude: Optional[List[str]] = None,
        is_pad_out: str = "is_pad",
        start_input_field: str = "start",
    ) -> None:

        assert enc_len > 0, "The value of `enc_len` should be > 0"
        assert dec_len > 0, "The value of `dec_len` should be > 0"

        self.train_sampler = train_sampler
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.target_field = target_field

        self.encoder_series_fields = (
            encoder_series_fields + [self.target_field]
            if encoder_series_fields is not None
            else [self.target_field]
        )
        self.decoder_series_fields = (
            decoder_series_fields + [self.target_field]
            if decoder_series_fields is not None
            else [self.target_field]
        )

        # Fields that are not used at prediction time for the decoder
        self.prediction_time_decoder_exclude = (
            prediction_time_decoder_exclude + [self.target_field]
            if prediction_time_decoder_exclude is not None
            else [self.target_field]
        )

        # Fields that are disabled for the decoder (dummy fields still created)
        self.decoder_disabled_fields = list(
            set(self.encoder_series_fields) - set(self.decoder_series_fields)
        )

        self.is_pad_out = is_pad_out
        self.start_in = start_input_field

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        target = data[self.target_field]

        if is_train:
            # We currently cannot handle time series that are shorter than the
            # prediction length during training, so we just skip these.
            # If we want to include them we would need to pad and to mask
            # the loss.
            if len(target) < self.dec_len:
                return

            sampling_indices = self.train_sampler(
                target, 0, len(target) - self.dec_len
            )
        else:
            sampling_indices = [len(target)]

        ts_fields_counter = Counter(
            set(self.encoder_series_fields + self.decoder_series_fields)
        )

        for sampling_idx in sampling_indices:
            # ensure start index is not negative
            start_idx = max(0, sampling_idx - self.enc_len)

            # irrelevant data should have been removed by now in the
            # transformation chain, so copying everything is ok
            out = data.copy()

            for ts_field in list(ts_fields_counter.keys()):

                # target is 1d, this ensures ts is always 2d
                ts = np.atleast_2d(out[ts_field])

                if ts_fields_counter[ts_field] == 1:
                    del out[ts_field]
                else:
                    ts_fields_counter[ts_field] -= 1

                # take enc_len values from ts, depending on sampling_idx
                slice = ts[:, start_idx:sampling_idx]

                # if we have less than enc_len values, pad_left with 0
                past_piece = pad_to_size(slice, self.enc_len)

                out[self._past(ts_field)] = past_piece.transpose()

                # exclude some fields at prediction time
                if (
                    not is_train
                    and ts_field in self.prediction_time_decoder_exclude
                ):
                    continue

                # This is were some of the forking magic happens:
                # For each of the encoder_len time-steps at which the decoder is applied we slice the
                # corresponding inputs called decoder_fields to the appropriate dec_len
                if (
                    ts_field
                    in self.decoder_series_fields
                    + self.decoder_disabled_fields
                ):
                    forking_dec_field = np.zeros(
                        shape=(self.enc_len, self.dec_len, len(ts))
                    )

                    # in case it's not disabled we copy the actual values
                    if ts_field not in self.decoder_disabled_fields:
                        skip = max(0, self.enc_len - sampling_idx)
                        # This section takes by far the longest time computationally:
                        # This scales linearly in self.enc_len and linearly in self.dec_len
                        for dec_field, idx in zip(
                            forking_dec_field[skip:],
                            range(start_idx + 1, start_idx + self.enc_len + 1),
                        ):
                            dec_field[:] = ts[:, idx : idx + self.dec_len].T

                    if forking_dec_field.shape[-1] == 1:
                        out[self._future(ts_field)] = np.squeeze(
                            forking_dec_field, axis=-1
                        )
                    else:
                        out[self._future(ts_field)] = forking_dec_field

            # So far pad indicator not in use
            pad_indicator = np.zeros(self.enc_len)
            pad_length = max(0, self.enc_len - sampling_idx)
            pad_indicator[:pad_length] = True
            out[self._past(self.is_pad_out)] = pad_indicator

            # So far pad forecast_start not in use
            out[FieldName.FORECAST_START] = shift_timestamp(
                out[self.start_in], sampling_idx
            )

            yield out

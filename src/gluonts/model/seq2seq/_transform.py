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
from typing import Iterator, List, Optional

# Third-party imports
import numpy as np
from numpy.lib.stride_tricks import as_strided

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import FlatMapTransformation, shift_timestamp


def pad_to_size(xs: np.array, size: int):
    """Pads `xs` with 0 on the left on the first axis."""
    pad_length = size - xs.shape[0]
    if pad_length <= 0:
        return xs

    pad_width = [(pad_length, 0)] + ([(0, 0)] * (xs.ndim - 1))
    return np.pad(xs, mode="constant", pad_width=pad_width)


class ForkingSequenceSplitter(FlatMapTransformation):
    """Forking sequence splitter."""

    @validated()
    def __init__(
        self,
        train_sampler,
        enc_len: int,
        dec_len: int,
        num_forking: Optional[int] = None,
        target_field: str = FieldName.TARGET,
        encoder_series_fields: Optional[List[str]] = None,
        decoder_series_fields: Optional[List[str]] = None,
        encoder_disabled_fields: Optional[List[str]] = None,
        decoder_disabled_fields: Optional[List[str]] = None,
        prediction_time_decoder_exclude: Optional[List[str]] = None,
        is_pad_out: str = "is_pad",
        start_input_field: str = "start",
    ) -> None:

        assert enc_len > 0, "The value of `enc_len` should be > 0"
        assert dec_len > 0, "The value of `dec_len` should be > 0"

        self.train_sampler = train_sampler
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.num_forking = (
            num_forking if num_forking is not None else self.enc_len
        )
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

        self.encoder_disabled_fields = (
            encoder_disabled_fields
            if encoder_disabled_fields is not None
            else []
        )

        self.decoder_disabled_fields = (
            decoder_disabled_fields
            if decoder_disabled_fields is not None
            else []
        )

        # Fields that are not used at prediction time for the decoder
        self.prediction_time_decoder_exclude = (
            prediction_time_decoder_exclude + [self.target_field]
            if prediction_time_decoder_exclude is not None
            else [self.target_field]
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

        # Loops over all encoder and decoder fields even those that are disabled to
        # set to dummy zero fields in those cases
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
                ts = np.atleast_2d(out[ts_field]).T

                if ts_fields_counter[ts_field] == 1:
                    del out[ts_field]
                else:
                    ts_fields_counter[ts_field] -= 1

                # take enc_len values from ts, depending on sampling_idx
                slice = ts[start_idx:sampling_idx, :]

                ts_len = ts.shape[1]
                past_piece = np.zeros(
                    shape=(self.enc_len, ts_len), dtype=ts.dtype
                )

                if ts_field not in self.encoder_disabled_fields:
                    # if we have less than enc_len values, pad_left with 0
                    past_piece = pad_to_size(slice, self.enc_len)
                out[self._past(ts_field)] = past_piece

                # exclude some fields at prediction time
                if (
                    not is_train
                    and ts_field in self.prediction_time_decoder_exclude
                ):
                    continue

                # This is were some of the forking magic happens:
                # For each of the encoder_len time-steps at which the decoder is applied we slice the
                # corresponding inputs called decoder_fields to the appropriate dec_len
                if ts_field in self.decoder_series_fields:

                    forking_dec_field = np.zeros(
                        shape=(self.num_forking, self.dec_len, ts_len),
                        dtype=ts.dtype,
                    )
                    # in case it's not disabled we copy the actual values
                    if ts_field not in self.decoder_disabled_fields:
                        # In case we sample and index too close to the beginning of the time series we would run out of
                        # bounds (i.e. try to copy non existent time series data) to prepare the input for the decoder.
                        # Instead of copying the partially available data from the time series and padding it with
                        # zeros, we simply skip copying the partial data. Since copying data would result in overriding
                        # the 0 pre-initialized 3D array, the end result of skipping is that the affected 2D decoder
                        # inputs (entries of the 3D array - of which there are skip many) will still be all 0."
                        skip = max(0, self.num_forking - sampling_idx)
                        start_idx = max(0, sampling_idx - self.num_forking)
                        # For 2D column-major (Fortran) ordering transposed array strides = (dtype, dtype*n_rows)
                        # For standard row-major arrays, strides = (dtype*n_cols, dtype)
                        stride = ts.strides
                        forking_dec_field[skip:, :, :] = as_strided(
                            ts[
                                start_idx
                                + 1 : start_idx
                                + 1
                                + self.num_forking
                                - skip,
                                :,
                            ],
                            shape=(
                                self.num_forking - skip,
                                self.dec_len,
                                ts_len,
                            ),
                            # strides for 2D array expanded to 3D array of shape (dim1, dim2, dim3) =
                            # strides for 2D array expanded to 3D array of shape (dim1, dim2, dim3) =
                            # (1, n_rows, n_cols).  Note since this array has been transposed, it is stored in
                            # column-major (Fortan) ordering, i.e. for transposed data of shape (dim1, dim2, dim3),
                            # strides = (dtype, dtype * dim1, dtype*dim1*dim2) = (dtype, dtype, dtype*n_rows).
                            strides=stride[0:1] + stride,
                        )
                    # edge case for prediction_length = 1
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

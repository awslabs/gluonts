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

from collections import Counter
from typing import Iterator, List, Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import FlatMapTransformation, shift_timestamp


class ForkingSequenceSplitter(FlatMapTransformation):
    """Forking sequence splitter."""

    @validated()
    def __init__(
        self,
        instance_sampler,
        enc_len: int,
        dec_len: int,
        num_forking: Optional[int] = None,
        target_field: str = FieldName.TARGET,
        encoder_series_fields: Optional[List[str]] = None,
        decoder_series_fields: Optional[List[str]] = None,
        encoder_disabled_fields: Optional[List[str]] = None,
        decoder_disabled_fields: Optional[List[str]] = None,
        is_pad_out: str = "is_pad",
        start_input_field: str = "start",
    ) -> None:

        assert enc_len > 0, "The value of `enc_len` should be > 0"
        assert dec_len > 0, "The value of `dec_len` should be > 0"

        self.instance_sampler = instance_sampler
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

        sampled_indices = self.instance_sampler(target)

        ts_fields = set(
            self.encoder_series_fields + self.decoder_series_fields
        )

        for idx in sampled_indices:
            # irrelevant data should have been removed by now in the
            # transformation chain, so copying everything is ok
            out = data.copy()

            enc_len_diff = idx - self.enc_len
            dec_len_diff = idx - self.num_forking

            # ensure start indices are not negative
            start_idx_enc = max(0, enc_len_diff)
            start_idx_dec = max(0, dec_len_diff)

            # Define pad length indices for shorter time series of variable length being updated in place
            pad_length_enc = max(0, -enc_len_diff)
            pad_length_dec = max(0, -dec_len_diff)

            for ts_field in ts_fields:

                # target is 1d, this ensures ts is always 2d
                ts = np.atleast_2d(out[ts_field]).T
                ts_len = ts.shape[1]

                del out[ts_field]

                out[self._past(ts_field)] = np.zeros(
                    shape=(self.enc_len, ts_len), dtype=ts.dtype
                )
                if ts_field not in self.encoder_disabled_fields:
                    out[self._past(ts_field)][pad_length_enc:] = ts[
                        start_idx_enc:idx, :
                    ]

                if ts_field in self.decoder_series_fields:
                    out[self._future(ts_field)] = np.zeros(
                        shape=(self.num_forking, self.dec_len, ts_len),
                        dtype=ts.dtype,
                    )
                    if ts_field not in self.decoder_disabled_fields:
                        # This is where some of the forking magic happens:
                        # For each of the num_forking time-steps at which the decoder is applied we slice the
                        # corresponding inputs called decoder_fields to the appropriate dec_len
                        decoder_fields = ts[start_idx_dec + 1 : idx + 1, :]
                        # For default row-major arrays, strides = (dtype*n_cols, dtype). Since this array is transposed,
                        # it is stored in column-major (Fortran) ordering with strides = (dtype, dtype*n_rows)
                        stride = decoder_fields.strides
                        out[self._future(ts_field)][
                            pad_length_dec:
                        ] = as_strided(
                            decoder_fields,
                            shape=(
                                self.num_forking - pad_length_dec,
                                self.dec_len,
                                ts_len,
                            ),
                            # strides for 2D array expanded to 3D array of shape (dim1, dim2, dim3) =
                            # (1, n_rows, n_cols).  For transposed data, strides =
                            # (dtype, dtype * dim1, dtype*dim1*dim2) = (dtype, dtype, dtype*n_rows).
                            strides=stride[0:1] + stride,
                        )

                    # edge case for prediction_length = 1
                    if out[self._future(ts_field)].shape[-1] == 1:
                        out[self._future(ts_field)] = np.squeeze(
                            out[self._future(ts_field)], axis=-1
                        )

            # So far encoder pad indicator not in use -
            # Marks that left padding for the encoder will occur on shorter time series
            pad_indicator = np.zeros(self.enc_len)
            pad_indicator[:pad_length_enc] = True
            out[self._past(self.is_pad_out)] = pad_indicator

            # So far pad forecast_start not in use
            out[FieldName.FORECAST_START] = shift_timestamp(
                out[self.start_in], idx
            )

            yield out

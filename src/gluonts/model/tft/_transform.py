from collections import Counter
from typing import Optional, List, Iterator

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import InstanceSplitter, shift_timestamp


class TFTInstanceSplitter(InstanceSplitter):
    @validated()
    def __init__(
        self,
        train_sampler,
        past_length: int,
        future_length: int,
        target_field: str = FieldName.TARGET,
        is_pad_field: str = FieldName.IS_PAD,
        start_field: str = FieldName.START,
        forecast_start_field: str = FieldName.FORECAST_START,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
        past_time_series_fields: Optional[List[str]] = None,
        pick_incomplete: bool = True,
        dummy_value: float = 0.0,
    ) -> None:

        assert past_length > 0, "The value of `past_length` should be > 0"
        assert future_length > 0, "The value of `future_length` should be > 0"

        self.train_sampler = train_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.pick_incomplete = pick_incomplete
        self.dummy_value = dummy_value

        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field

        self.ts_fields = time_series_fields or []
        self.past_ts_fields = past_time_series_fields or []

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pl = self.future_length
        lt = self.lead_time
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]
        len_target = len_target = target.shape[-1]

        minimum_length = (
            self.future_length
            if self.pick_incomplete
            else self.past_length + self.future_length
        ) + self.lead_time

        if is_train:
            sampling_bounds = (
                (
                    0,
                    len_target - self.future_length - self.lead_time,
                )  # TODO: create parameter lower sampling bound for NBEATS
                if self.pick_incomplete
                else (
                    self.past_length,
                    len_target - self.future_length - self.lead_time,
                )
            )

            # We currently cannot handle time series that are
            # too short during training, so we just skip these.
            # If we want to include them we would need to pad and to
            # mask the loss.
            sampled_indices = (
                np.array([], dtype=int)
                if len_target < minimum_length
                else self.train_sampler(target, *sampling_bounds)
            )
        else:
            assert self.pick_incomplete or len_target >= self.past_length
            sampled_indices = np.array([len_target], dtype=int)

        for i in sampling_indices:
            pad_length = max(self.past_length - i, 0)
            if not self.pick_incomplete:
                assert (
                    pad_length == 0
                ), f"pad_length should be zero, got {pad_length}"
            d = data.copy()
            for ts_field in slice_cols:
                if i >= self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length : i]
                else:
                    pad_block = (
                        np.ones(
                            d[ts_field].shape[:-1] + (pad_length,),
                            dtype=d[ts_field].dtype,
                        )
                        * self.dummy_value
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                d[self._past(ts_field)] = past_piece
                if ts_field not in self.past_ts_fields:
                    d[self._future(ts_field)] = d[ts_field][
                        ..., (i + lt) : (i + lt + pl)
                    ]
                del d[ts_field]
            pad_indicator = np.zeros(self.past_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.output_NTC:
                for ts_field in slice_cols:
                    d[self._past(ts_field)] = d[
                        self._past(ts_field)
                    ].transpose()
                    d[self._future(ts_field)] = d[
                        self._future(ts_field)
                    ].transpose()

            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], i + lt
            )
            yield d

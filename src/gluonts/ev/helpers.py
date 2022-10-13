from typing import Collection, Dict, Iterator, Optional

import numpy as np

from ..model.forecast import Forecast
from ..dataset.split import TestData


EvalData = Dict[str, np.ndarray]


def axis_is_zero_or_none(axis: Optional[int]) -> bool:
    return axis == 0 or axis is None


def gather_inputs(
    test_data: TestData,
    forecasts: Iterator[Forecast],
    quantile_levels: Collection[float],
    batch_size: int = 64,
) -> Iterator[EvalData]:
    """Collect relevant data as NumPy arrays to evaluate the next batch.

    The number of entries in `test_data` and `forecasts` must be equal.
    """
    inputs = iter(test_data.input)
    labels = iter(test_data.label)

    def get_empty_eval_data():
        input_data = []
        label_data = []
        forecast_data = {
            "mean": [],
            **{str(q): [] for q in quantile_levels},
        }
        return input_data, label_data, forecast_data

    def make_data(input_data, label_data, forecast_data):
        return {
            "input": np.stack([entry["target"] for entry in input_data]),
            "label": np.stack([entry["target"] for entry in label_data]),
            **{name: np.stack(value) for name, value in forecast_data.items()}
        }

    input_data, label_data, forecast_data = get_empty_eval_data()

    entry_counter = 0
    for _ in range(len(test_data)):
        forecast_entry = next(forecasts)
        forecast_data["mean"].append(forecast_entry.mean)
        for q in quantile_levels:
            forecast_data[str(q)].append(forecast_entry.quantile(q))

        input_data.append(next(inputs))
        label_data.append(next(labels))

        entry_counter += 1
        if entry_counter % batch_size == 0:
            yield make_data(input_data, label_data, forecast_data)

            input_data, label_data, forecast_data = get_empty_eval_data()

    if entry_counter % batch_size != 0:
       yield make_data(input_data, label_data, forecast_data)


class DataProbe:
    """ A DataProbe gathers all quantile forecasts required for an evaluation.
    This has the benefit that metric definitions can work independently of
    `Forecast` objects as all values in 'data' will be NumPy arrays.
    :raises ValueError: if a metric requests a key that can't be converted to
        float and isn't equal to "batch_size", "input", "label" or "mean"
    """
    def __init__(self, test_data: TestData):
        input_sample, label_sample = next(iter(test_data))
        # use batch_size 1
        self.input_shape = (1,) + np.shape(input_sample["target"])
        self.prediction_target_shape = (1,) + np.shape(label_sample["target"])

        self.required_quantile_forecasts = set()

    def __getitem__(self, key: str):
        if key == "batch_size":
            return 1
        if key == "input":
            return np.random.rand(*self.input_shape)
        if key in ["label", "mean"]:
            return np.random.rand(*self.prediction_target_shape)

        try:
            self.required_quantile_forecasts.add(float(key))
            return np.random.rand(*self.prediction_target_shape)
        except ValueError:
            raise ValueError(f"Unexpected input: {key}")
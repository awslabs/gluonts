import numpy as np


def abs_label(data: dict):
    return np.abs(data["label"])


def error(data: dict, forecast_type: str):
    return data["label"] - data[forecast_type]


def abs_error(data: dict, forecast_type: str):
    return np.abs(error(data, forecast_type))


def squared_error(data: dict, forecast_type: str):
    return np.square(error(data, forecast_type))


def quantile_loss(data: dict, q: float):
    forecast_type = str(q)
    prediction = data[forecast_type]

    return np.abs(
        error(data, forecast_type) * ((prediction >= data["label"]) - q)
    )


def coverage(data: dict, q: float):
    forecast_type = str(q)
    return data["label"] < data[forecast_type]


def absolute_percentage_error(data: dict, forecast_type: str = "0.5"):
    return abs_error(data, forecast_type) / abs_label(data)


def symmetric_absolute_percentage_error(
    data: dict, forecast_type: str = "0.5"
):
    return abs_error(data, forecast_type) / (
        abs_label(data) + np.abs(data[forecast_type])
    )

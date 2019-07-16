# Third-party imports
import numpy as np
import pytest

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.model.prophet import ProphetPredictor, PROPHET_IS_INSTALLED

# conditionally skip these tests if `fbprophet` is not installed
# see https://docs.pytest.org/en/latest/skipping.html for details
if not PROPHET_IS_INSTALLED:
    skip_message = 'Skipping test because `fbprophet` is not installed'
    pytest.skip(msg=skip_message, allow_module_level=True)


def test_feat_dynamic_real_success():
    params = dict(
        freq="1D", prediction_length=3, prophet_params=dict(n_changepoints=20)
    )

    dataset = ListDataset(
        data_iter=[
            {
                "start": "2017-01-01",
                "target": np.array([1.0, 2.0, 3.0, 4.0]),
                "feat_dynamic_real": np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    ]
                ),
            }
        ],
        freq=params["freq"],
    )

    predictor = ProphetPredictor(**params)

    act_fcst = next(predictor.predict(dataset))
    exp_fcst = np.arange(5.0, 5.0 + params["prediction_length"])

    assert np.all(np.isclose(act_fcst.quantile(0.1), exp_fcst, atol=0.02))
    assert np.all(np.isclose(act_fcst.quantile(0.5), exp_fcst, atol=0.02))
    assert np.all(np.isclose(act_fcst.quantile(0.9), exp_fcst, atol=0.02))


def test_feat_dynamic_real_bad_size():
    params = dict(freq="1D", prediction_length=3, prophet_params={})

    dataset = ListDataset(
        data_iter=[
            {
                "start": "2017-01-01",
                "target": np.array([1.0, 2.0, 3.0, 4.0]),
                "feat_dynamic_real": np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ]
                ),
            }
        ],
        freq=params["freq"],
    )

    with pytest.raises(AssertionError) as excinfo:
        predictor = ProphetPredictor(**params)
        list(predictor.predict(dataset))

    assert str(excinfo.value) == (
        "Length mismatch for dynamic real-valued feature #0: "
        "expected 7, got 6"
    )


def test_min_obs_error():
    params = dict(freq="1D", prediction_length=10, prophet_params={})

    dataset = ListDataset(
        data_iter=[{"start": "2017-01-01", "target": np.array([1.0])}],
        freq=params["freq"],
    )

    with pytest.raises(ValueError) as excinfo:
        predictor = ProphetPredictor(**params)
        list(predictor.predict(dataset))

    act_error_msg = str(excinfo.value)
    exp_error_msg = "Dataframe has less than 2 non-NaN rows."

    assert act_error_msg == exp_error_msg

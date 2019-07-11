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


def test_related_time_series_success():
    params = dict(
        freq="1D", prediction_length=3, prophet_params=dict(n_changepoints=20)
    )

    dataset = ListDataset(
        data_iter=[
            {
                'start': '2017-01-01',
                'target': np.array([1.0, 2.0, 3.0, 4.0]),
                'feat_dynamic_real': np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    ]
                ),
            }
        ],
        freq=params['freq'],
    )

    predictor = ProphetPredictor(**params)
    list(predictor.predict(dataset))


def test_related_time_series_fail():
    params = dict(freq="1D", prediction_length=3, prophet_params={})

    dataset = ListDataset(
        data_iter=[
            {
                'start': '2017-01-01',
                'target': np.array([1.0, 2.0, 3.0, 4.0]),
                'feat_dynamic_real': np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ]
                ),
            }
        ],
        freq=params['freq'],
    )

    with pytest.raises(AssertionError) as excinfo:
        predictor = ProphetPredictor(**params)
        list(predictor.predict(dataset))

    act_error_msg = str(excinfo.value)
    exp_error_msg = (
        'Length mismatch for dynamic real-valued feature #0: '
        'expected 7, got 6'
    )

    assert act_error_msg == exp_error_msg


def test_min_obs():
    params = dict(freq="1D", prediction_length=10, prophet_params={})

    dataset = ListDataset(
        data_iter=[{'start': '2017-01-01', 'target': np.array([1.0])}],
        freq=params['freq'],
    )

    with pytest.raises(ValueError) as excinfo:
        predictor = ProphetPredictor(**params)
        list(predictor.predict(dataset))

    act_error_msg = str(excinfo.value)
    exp_error_msg = 'Dataframe has less than 2 non-NaN rows.'

    assert act_error_msg == exp_error_msg


def test_min_obs_with_nans():
    params = dict(freq="1D", prediction_length=10, prophet_params={})

    dataset = ListDataset(
        data_iter=[
            {'start': '2017-01-01', 'target': np.array([1.0, "nan", "nan"])}
        ],
        freq=params['freq'],
    )

    with pytest.raises(ValueError) as excinfo:
        predictor = ProphetPredictor(**params)
        list(predictor.predict(dataset))

    act_error_msg = str(excinfo.value)
    exp_error_msg = 'Dataframe has less than 2 non-NaN rows.'

    assert act_error_msg == exp_error_msg

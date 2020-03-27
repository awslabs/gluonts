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

# Third-party imports
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.core.exception import GluonTSDataError
from gluonts.dataset.common import Dataset, ListDataset, DataEntry
from gluonts.model.npts import KernelType, NPTSPredictor
from gluonts.model.npts._weighted_sampler import WeightedSampler


def get_test_data(history_length: int, freq: str) -> pd.Series:
    index = pd.date_range("1/1/2011", periods=history_length, freq=freq)
    return pd.Series(np.arange(len(index)), index=index)


@pytest.mark.parametrize(
    "freq, history_length, num_seasons",
    [
        ("min", 480, 60),
        ("15min", 96, 4),
        ("H", 336, 24),
        ("4H", 336, 6),
        ("D", 56, 7),
        ("M", 96, 12),
        ("3M", 32, 4),
        ("12M", 8, 1),
    ],
)
@pytest.mark.parametrize("use_seasonal_model", [False, True])
@pytest.mark.parametrize("context_length_frac", [None, 0.5, 2.0])
def test_climatological_forecaster(
    freq: str,
    history_length: int,
    num_seasons: int,
    use_seasonal_model: bool,
    context_length_frac: float,
) -> None:
    """
    Here we test Climatological forecaster both seasonal and non-seasonal
    variants for various frequencies.

    We further parametrize the test with `context_length_frac` parameter which
    indicates what fraction of the history should actually be used including
    the case where `context_length` is larger than the history length.

    In particular, we check that:

    1. predictions are uniformly sampled over the entire past for non-seasonal
       variant and only over the past seasons for the seasonal variant;
    2. predictions do not come from the targets outsides of the
       `context_length`, if `context_length` is present.

    Parameters
    ----------
    freq
        frequency of the time series
    history_length
        length of the time series to be generated for tests
    num_seasons
        number of seasons present in a given frequency
    use_seasonal_model
        use the seasonal variant?
    context_length_frac
        fraction of history length that should be used as context length
    """

    train_ts = get_test_data(history_length=history_length, freq=freq)

    # For seasonal variant we check the predictions of Climatological
    # forecaster for all seasons.
    # Non-seasonal variant is modeled in the tests as `num_seasons = 1`, and
    # the predictions are checked for only one step.
    num_seasons = num_seasons if use_seasonal_model else 1
    pred_length = num_seasons

    context_length = (
        int(context_length_frac * history_length)
        if context_length_frac is not None
        else None
    )
    predictor = NPTSPredictor(
        prediction_length=pred_length,
        context_length=context_length,
        freq=freq,
        use_seasonal_model=use_seasonal_model,
        kernel_type=KernelType.uniform,
    )

    dataset = ListDataset(
        [{"start": str(train_ts.index[0]), "target": train_ts.values}],
        freq=freq,
    )

    # validate that the predictor works with targets with NaNs
    _test_nans_in_target(predictor, dataset)

    forecast = next(predictor.predict(dataset, num_samples=2000))

    train_targets = (
        train_ts.values
        if context_length is None
        else train_ts.values[-min(history_length, context_length) :]
    )
    targets_outside_context = (
        None
        if context_length is None or context_length >= history_length
        else train_ts.values[:context_length]
    )
    targets_str = "seasons" if use_seasonal_model else "targets"
    seasonal_str = "seasonal" if use_seasonal_model else ""
    for t in range(pred_length):
        targets_prev_seasons = train_targets[
            range(t, len(train_targets), num_seasons)
        ]

        # Predictions must have come from the past seasons only
        assert set(forecast.samples[:, t]).issubset(targets_prev_seasons), (
            f"Predictions for {seasonal_str} Climatological forecaster are "
            f"not generated from the target values of past {targets_str}.\n"
            f"Past {targets_str}: {targets_prev_seasons}\n"
            f"Predictions: {set(forecast.samples[:, t])}"
        )

        # Predictions must have been uniformly sampled from the targets in
        # the previous seasons
        prediction_dist, _ = np.histogram(
            forecast.samples[:, t], np.append(targets_prev_seasons, np.inf)
        )
        prediction_dist = prediction_dist / sum(prediction_dist)

        expected_dist = np.ones_like(targets_prev_seasons) / len(
            targets_prev_seasons
        )

        np.testing.assert_almost_equal(
            prediction_dist,
            expected_dist,
            1,
            f"Predictions of {seasonal_str} Climatological forecaster are not "
            f"uniformly sampled from past {targets_str}\n"
            f"Expected distribution over the past "
            f"{targets_str}: {expected_dist}\n"
            f"Prediction distribution: {prediction_dist}\n",
        )

        if targets_outside_context is not None:
            # Predictions should never be from the targets outside the
            # context length
            assert not set.intersection(
                set(forecast.samples[:, t]), set(targets_outside_context)
            ), (
                f"Predictions of Climatological forecaster are sampled from "
                f"targets outside the context length.\n"
                f"Targets outside the context length: "
                f"{targets_outside_context}\n"
                f"Predictions: {set(forecast.samples[:, t])}\n"
            )


@pytest.mark.parametrize(
    "freq, history_length, num_seasons",
    [
        ("min", 360, 60),
        ("15min", 96, 4),
        ("H", 336, 24),
        ("4H", 336, 6),
        ("D", 56, 7),
        ("M", 96, 12),
        ("3M", 32, 4),
        ("12M", 8, 1),
    ],
)
@pytest.mark.parametrize("use_seasonal_model", [False, True])
@pytest.mark.parametrize(
    "feature_scale, min_frac_samples_from_seasons",
    [(1.0, 0.0), (1000.0, 0.99)],
)
@pytest.mark.parametrize("context_length_frac", [None, 0.5])
def test_npts_forecaster(
    freq: str,
    history_length: int,
    num_seasons: int,
    use_seasonal_model: bool,
    feature_scale: float,
    min_frac_samples_from_seasons: float,
    context_length_frac: Optional[float],
) -> None:
    """
    Here we test both seasonal (num_seasons=24) and non-seasonal
    (num_seasons=1) variants of NPTS for various frequencies.

    We further parametrize the test with `context_length_frac` parameter which
    indicates what fraction of the history should actually be used.

    In particular, we check that the

    1. the predictions must come from past seasons exclusively with high
       probability for large value of `feature_scale`

    2. predictions are sampled according to exponentially decaying weights over
       the targets from past seasons or whole of training history depending on
       the flag `use_seasonal_model`

    3. predictions are sampled from time points that are not seasons as well
       for small value of `feature_scale`

    4. predictions do not come from the targets outsides of the context length,
       if `context_length` is present.

    Parameters
    ----------
    freq
        frequency of the time series
    history_length
        length of the time series to be generated for tests
    num_seasons
        number of seasons present in a given frequency
    use_seasonal_model
        use the seasonal variant?
    feature_scale
        scale for the seasonal features to enforce strict sampling over the
        past seasons
    min_frac_samples_from_seasons
        the minimum threshold for fraction of times the predictions should come
        exclusively from past seasons
    context_length_frac
        fraction of history length that should be used as context length
    """

    train_ts = get_test_data(history_length=history_length, freq=freq)

    # For seasonal variant we check the predictions of NPTS forecaster for
    # all seasons.
    # Non-seasonal variant is modeled in the tests as `num_seasons = 1`, and
    # the predictions are checked for only one step.
    num_seasons = num_seasons if use_seasonal_model else 1
    pred_length = num_seasons

    context_length = (
        int(context_length_frac * history_length)
        if context_length_frac is not None
        else None
    )
    predictor = NPTSPredictor(
        prediction_length=pred_length,
        context_length=context_length,
        freq=freq,
        kernel_type=KernelType.exponential,
        feature_scale=feature_scale,
        use_seasonal_model=use_seasonal_model,
    )

    dataset = ListDataset(
        [{"start": str(train_ts.index[0]), "target": train_ts.values}],
        freq=freq,
    )

    # validate that the predictor works with targets with NaNs
    _test_nans_in_target(predictor, dataset)

    forecast = next(predictor.predict(dataset, num_samples=2000))

    train_targets = (
        train_ts.values
        if context_length is None
        else train_ts.values[-min(history_length, context_length) :]
    )
    targets_outside_context = (
        None
        if context_length is None or context_length >= history_length
        else train_ts.values[:context_length]
    )
    targets_str = "seasons" if use_seasonal_model else "targets"
    seasonal_str = "seasonal" if use_seasonal_model else ""
    for t in range(pred_length):
        prev_seasons_ix = range(t, len(train_targets), num_seasons)

        # Prediction distribution over all the training targets
        prediction_dist, _ = np.histogram(
            forecast.samples[:, t], np.append(train_targets, np.inf)
        )
        prediction_dist = prediction_dist / sum(prediction_dist)

        # The fraction of times the targets from past seasons are sampled
        # exclusively should be above some threshold.
        assert (
            sum(prediction_dist[prev_seasons_ix])
            > min_frac_samples_from_seasons
        ), (
            f"Predictions of {seasonal_str} NPTS are not sampled from past "
            f"{targets_str} enough number of times.\n"
            f"Expected frequency over past {targets_str}: "
            f"{min_frac_samples_from_seasons}\n"
            f"Sampled frequency: {sum(prediction_dist[prev_seasons_ix])}"
        )

        if feature_scale == 1.0:
            # Time index feature and seasonal features are given equal
            # importance so we expect to see some predictions
            # coming outside of past seasons.
            non_seasons_ix = list(
                set(range(len(train_targets))) - set(prev_seasons_ix)
            )
            if non_seasons_ix:
                assert sum(prediction_dist[non_seasons_ix]) > 0.0, (
                    "Predictions of {seasonal_str} NPTS are expected to come "
                    "from targets not in the previous seasons"
                    "for small value of feature_scale: {feature_scale}"
                )

        if feature_scale == 1000.0:
            # Here we sample mostly from past seasons. In this case, the past
            # seasons must be sampled with exponentially
            # decaying weights which depend only on the time index
            # feature: f(t) = t / (train_length + pred_length)
            distance_to_prev_seasons = np.arange(
                len(train_targets) + 1, 1, -num_seasons
            ) / (len(train_targets) + pred_length)
            expected_dist_seasons = np.exp(-distance_to_prev_seasons)
            expected_dist_seasons /= sum(expected_dist_seasons)

            prediction_dist_seasons = prediction_dist[prev_seasons_ix]
            prediction_dist_seasons /= sum(prediction_dist_seasons)

            np.testing.assert_almost_equal(
                prediction_dist_seasons,
                expected_dist_seasons,
                1,
                f"Predictions of {seasonal_str} NPTS are not sampled with "
                f"exponentially decaying weights over the past "
                f"{targets_str}.\nExpected distribution over the past "
                f"{targets_str}: {expected_dist_seasons}\n"
                f"Prediction_dist: {prediction_dist_seasons}",
            )

        if targets_outside_context is not None:
            # Predictions should never be from the targets outside the context
            # length
            assert not set.intersection(
                set(forecast.samples[:, t]), set(targets_outside_context)
            ), (
                "Predictions of NPTS forecaster are sampled from targets "
                "outside the context length.\n"
                f"Targets outside the context length: "
                f"{targets_outside_context}\n"
                f"Predictions: {set(forecast.samples[:, t])}"
            )


@pytest.mark.parametrize("use_seasonal_model", [False, True])
@pytest.mark.parametrize(
    "feature_scale, min_frac_samples_from_seasons", [(1000.0, 0.99)]
)
@pytest.mark.parametrize("context_length_frac", [None, 0.5])
def test_npts_custom_features(
    use_seasonal_model: bool,
    feature_scale: float,
    min_frac_samples_from_seasons: float,
    context_length_frac: Optional[float],
) -> None:
    """
    Same as `test_npts_forecaster` except that we use the weekly frequency and
    a dummy custom feature to define seasonality. The dummy feature defines 52
    weeks as one cycle.

    We explicitly disable `use_default_time_features` so that the seasonality
    is defined based only on the custom feature.

    Parameters
    ----------
    use_seasonal_model
        use the seasonal variant?
    feature_scale
        scale for the seasonal features to enforce strict sampling over the
        past seasons
    min_frac_samples_from_seasons
        the minimum threshold for fraction of times the predictions should come
        exclusively from past seasons
    context_length_frac
        fraction of history length that should be used as context length
    """
    freq = "W"
    history_length = 52 * 8  # approx. 8 years (seasons)
    train_ts = get_test_data(history_length=history_length, freq=freq)
    context_length = (
        int(context_length_frac * history_length)
        if context_length_frac is not None
        else None
    )

    num_seasons = 52 if use_seasonal_model else 1
    pred_length = num_seasons

    # Custom features should be defined both in training and prediction ranges
    full_time_index = pd.date_range(
        train_ts.index.min(),
        periods=len(train_ts) + pred_length,
        freq=train_ts.index.freq,
    )
    # Dummy feature defining 52 seasons
    feat_dynamic_real = [
        [
            (ix % 52) / 51.0 - 0.5
            for ix, timestamp in enumerate(full_time_index)
        ]
    ]

    predictor = NPTSPredictor(
        prediction_length=pred_length,
        freq=freq,
        context_length=context_length,
        kernel_type=KernelType.exponential,
        feature_scale=feature_scale,
        use_seasonal_model=use_seasonal_model,
        use_default_time_features=False,  # disable default time features
    )

    dataset = ListDataset(
        [
            {
                "start": str(train_ts.index[0]),
                "target": train_ts.values,
                "feat_dynamic_real": feat_dynamic_real,
            }
        ],
        freq=freq,
    )

    # validate that the predictor works with targets with NaNs
    _test_nans_in_target(predictor, dataset)

    forecast = next(predictor.predict(dataset, num_samples=2000))

    train_targets = (
        train_ts.values
        if context_length is None
        else train_ts.values[-min(history_length, context_length) :]
    )
    targets_outside_context = (
        None
        if context_length is None or context_length >= history_length
        else train_ts.values[:context_length]
    )
    targets_str = "seasons" if use_seasonal_model else "targets"
    seasonal_str = "seasonal" if use_seasonal_model else ""
    for t in range(pred_length):
        prev_seasons_ix = range(t, len(train_targets), num_seasons)

        # Prediction distribution over all the training targets
        prediction_dist, _ = np.histogram(
            forecast.samples[:, t], np.append(train_targets, np.inf)
        )
        prediction_dist = prediction_dist / sum(prediction_dist)

        # The fraction of times the targets from past seasons are sampled
        # exclusively should be above some threshold.
        assert (
            sum(prediction_dist[prev_seasons_ix])
            > min_frac_samples_from_seasons
        ), (
            f"Predictions of {seasonal_str} NPTS are not sampled from past "
            f"{targets_str} enough number of times.\n"
            f"Expected frequency over past {targets_str}: "
            f"{min_frac_samples_from_seasons}\n"
            f"Sampled frequency: {sum(prediction_dist[prev_seasons_ix])}"
        )

        # Here we use large value of `feature_scale`, so we sample mostly
        # from past seasons. In this case, the past seasons must be sampled
        # with exponentially decaying weights which depend only
        # on the time index feature: f(t) = t / (train_length + pred_length)
        distance_to_prev_seasons = np.arange(
            len(train_targets) + 1, 1, -num_seasons
        ) / (len(train_targets) + pred_length)
        expected_dist_seasons = np.exp(-distance_to_prev_seasons)
        expected_dist_seasons /= sum(expected_dist_seasons)

        prediction_dist_seasons = prediction_dist[prev_seasons_ix]
        prediction_dist_seasons /= sum(prediction_dist_seasons)

        np.testing.assert_almost_equal(
            expected_dist_seasons,
            prediction_dist_seasons,
            1,
            f"Predictions of {seasonal_str} NPTS are not sampled with "
            f"exponentially decaying weights over the "
            f"past {targets_str}.\nExpected distribution over the past "
            f"{targets_str}: {expected_dist_seasons}\n"
            f"Prediction_dist: {prediction_dist_seasons}",
        )

        if targets_outside_context is not None:
            # Predictions should never be from the targets outside the context
            # length
            assert not set.intersection(
                set(forecast.samples[:, t]), set(targets_outside_context)
            ), (
                f"Predictions of NPTS forecaster are sampled from targets"
                f"outside the context length.\n"
                f"Targets outside the context length:"
                f"{targets_outside_context}\n"
                f"Predictions: {set(forecast.samples[:, t])}"
            )


def _test_nans_in_target(predictor: NPTSPredictor, dataset: Dataset) -> None:
    """
    Test that the model behaves as expected when the target time series
    contains NaN values.

    Parameters
    ----------
    predictor
        the predictor instance to test
    dataset
        a dataset (with targets without NaNs) to use as a base for the test
    """

    # a copy of dataset with 90% of the target entries NaNs
    ds_090pct_nans = ListDataset(
        data_iter=[
            _inject_nans_in_target(data_entry, p=0.9) for data_entry in dataset
        ],
        freq=predictor.freq,
    )

    # a copy of dataset with 100% of the target entries NaNs
    ds_100pct_nans = ListDataset(
        data_iter=[
            _inject_nans_in_target(data_entry, p=1.0) for data_entry in dataset
        ],
        freq=predictor.freq,
    )

    # assert that we can tolerate a high percentage of NaNs
    for forecast in predictor.predict(ds_090pct_nans):
        assert np.all(np.isfinite(forecast.samples)), "Forecast contains NaNs."

    # assert that an exception is thrown if 100% of the values are NaN
    with pytest.raises(GluonTSDataError) as excinfo:
        for _ in predictor.predict(ds_100pct_nans):
            pass
    assert (
        f"The last {predictor.context_length} positions of the target time "
        f"series are all NaN. Please increase the `context_length` "
        f"parameter of your NPTS model so the last "
        f"{predictor.context_length} positions of each target contain at "
        f"least one non-NaN value."
    ) in str(excinfo.value)


def _inject_nans_in_target(data_entry: DataEntry, p: float) -> DataEntry:
    """
    Returns a copy of the given `data_entry` where approximately `p` percent
    of the target values are NaNs.

    Parameters
    ----------
    data_entry
        The data entry to use as source.
    p
        The fraction of target positions to set to NaN (between 0 and 1).

    Returns
    -------
        A copy of `data_entry` with modified target field.
    """
    nan_positions = np.sort(
        a=np.random.choice(
            a=np.arange(data_entry["target"].size, dtype=np.int),
            size=int(p * data_entry["target"].size),
            replace=False,
        )
    )

    nan_target = np.copy(data_entry["target"])
    nan_target[nan_positions] = np.nan

    # if p < 1.0 at the last position should be kept unchanged
    # otherwise for large p we might end up with NaNs in the last
    # context_length positions
    if p < 1.0:
        nan_target[-1] = data_entry["target"][-1]

    return {
        key: (nan_target if key == "target" else val)
        for key, val in data_entry.items()
    }


@pytest.mark.parametrize("frac_zero_weights", [0.0, 0.25, 0.99, 1.0])
def test_weighted_sampler(frac_zero_weights: float) -> None:
    """
    We test the weighted sampler with weights that have varying fraction of
    zeros where positions for zero are randomly selected. Indices/Positions
    with zero weight should never be sampled.

    For the special case `frac_zero_weights = 1.0`, we make all the weights
    zero and expect that each of the indices is sampled with equal probability.

    Parameters
    ----------
    frac_zero_weights
        fraction of weights that are zeros.
    """

    # random unnormalized weights
    num_weights = 1000
    weights = np.random.random(num_weights)

    # randomly make some fraction of the weights zeros to test that they are
    # never sampled.
    if frac_zero_weights != 1.0:
        zeros_ix = np.random.randint(
            low=0, high=num_weights, size=int(num_weights * frac_zero_weights)
        )
    else:
        # For the special case `frac_zero_weights` we select all indices.
        zeros_ix = np.arange(num_weights)
    weights[zeros_ix] = 0.0

    num_samples = 100_000
    samples_ix = WeightedSampler.sample(weights, num_samples)

    # empirical probabilities for each index
    counts_ix, _ = np.histogram(samples_ix, bins=range(num_weights + 1))
    probs_ix = counts_ix / sum(counts_ix)

    true_prob_ix = (
        weights / sum(weights)
        if frac_zero_weights != 1.0
        else np.ones_like(weights) / num_weights
    )

    np.testing.assert_almost_equal(
        probs_ix,
        true_prob_ix,
        2,
        "Empirical distribution does not match sampling distribution",
    )

    if frac_zero_weights != 1.0:
        assert all(
            probs_ix[zeros_ix] == 0.0
        ), "Indices with sampling weight zero are sampled!"

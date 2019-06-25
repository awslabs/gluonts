# Standard library imports
from typing import Callable, Iterator, List, cast

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.model.forecast import SampleForecast

# Relative imports
from ._weighted_sampler import WeightedSampler


class NPTS:
    """
    Here we collect all the methods needed for generating NPTS Forecasts.
    """

    @staticmethod
    def compute_weights(
        train_features: np.ndarray,
        pred_features: np.ndarray,
        target_isnan_positions: np.ndarray,
        kernel: Callable[[np.ndarray, np.ndarray], float],
        do_exp: bool = True,
    ) -> Iterator[np.ndarray]:
        """
        Given the (logarithm of) kernel as well as training and prediction
        range features, this method returns an iterator over sampling weights
        for each time step in the prediction range.

        Note that the number of sampling weights for each time step vary since
        the prediction for time step `pred_t` samples from all the training
        targets as well as predictions until `pred_t` - 1.

        :return: iterator over sampling weights

        Parameters
        ----------
        train_features
            shape: (num_features, train_length)
        pred_features
            shape: (num_features, prediction_length)
        target_isnan_positions:
            an array of indices where the target is a NaN
        kernel
            kernel function that maps pairs of arrays to real numbers
        do_exp:
            exponentiate the weights in case of exponential kernel
            (for numerical stability we do this here)

        Returns
        -------
        iterator over sampling weights
        """

        assert len(np.shape(train_features)) == 2, (
            "Train features should be 2D-array where the rows represent "
            "features and columns the time points."
        )

        assert len(np.shape(pred_features)) == 2, (
            "Prediction features should be 2D-array where the rows represent "
            "features and columns the time points."
        )

        train_length = train_features.shape[1]
        prediction_length = pred_features.shape[1]

        for pred_t in range(prediction_length):
            # Prediction for `pred_t` samples from all the training targets
            # as well as predictions until `pred_t` - 1
            sampling_weights = np.zeros(train_length + pred_t)
            for train_t in range(train_length):
                sampling_weights[train_t] = kernel(
                    train_features[:, train_t], pred_features[:, pred_t]
                )

            for t in range(pred_t):
                sampling_weights[train_length + t] = kernel(
                    pred_features[:, t], pred_features[:, pred_t]
                )

            if do_exp:
                # To avoid numerical issues with exponentiation.
                sampling_weights -= max(sampling_weights)
                sampling_weights = np.exp(sampling_weights)

            # reset kernel at positions where the target is NaN
            sampling_weights[target_isnan_positions] = 0.0

            # Sometimes (e.g. for a for seasonal climatological kernel ) all
            # positions with non-zero probability are NaNs, so after resetting
            # the weights at these positions sampling_weights has only zeroes.
            # In this case, we want to sample uniformly from the observed
            # positions.
            if np.sum(sampling_weights) == 0:
                sampling_weights[target_isnan_positions] = -1.0
                sampling_weights += 1.0

            yield sampling_weights

    @staticmethod
    def predict(
        targets: pd.Series,
        prediction_length: int,
        sampling_weights_iterator: Iterator[np.ndarray],
        num_samples: int,
    ) -> SampleForecast:
        """
        Given the `targets`, generates `Forecast` containing prediction
        samples for `predcition_length` time points.

        Predictions are generated via weighted sampling where the weights are
        specified in `sampling_weights_iterator`.

        Parameters
        ----------
        targets
            targets to predict
        prediction_length
            prediction length
        sampling_weights_iterator
            iterator over weights used for sampling
        num_samples
            number of samples to set in the :class:`SampleForecast` object

        Returns
        -------
        SampleForecast
           a :class:`SampleForecast` object for the given targets
        """

        # Note that to generate prediction from the second time step onwards,
        # we need the sample predicted for all the previous time steps in the
        # prediction range.
        # To make this simpler, we replicate the training targets for
        # `num_samples` times.

        # samples shape: (num_samples, train_length + prediction_length)
        samples = np.tile(
            A=np.concatenate((targets.values, np.zeros(prediction_length))),
            reps=(num_samples, 1),
        )

        train_length = len(targets)
        for t, sampling_weights in enumerate(sampling_weights_iterator):
            samples_ix = WeightedSampler.sample(sampling_weights, num_samples)
            samples[:, train_length + t] = samples[
                np.arange(num_samples), samples_ix
            ]

        # Forecast takes as input the prediction range samples, the start date
        # of the prediction range, and the frequency of the time series.
        samples_pred_range = samples[:, train_length:]  # prediction range only
        forecast_start = targets.index[-1] + 1

        return SampleForecast(
            samples=samples_pred_range,
            start_date=forecast_start,
            freq=forecast_start.freqstr,
        )

    @staticmethod
    def log_distance_kernel(
        alpha: float
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda x, y: cast(float, -alpha * np.sum(np.abs(x - y)))

    @staticmethod
    def log_weighted_distance_kernel(
        kernel_weights: List[float]
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        kernel_weights_nd = np.ndarray(kernel_weights, dtype=np.float32)
        return lambda x, y: cast(
            float, -np.sum(kernel_weights_nd * np.abs(x - y))
        )

    @staticmethod
    def uniform_kernel() -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda x, y: 1.0 if np.sum(np.abs(x - y)) == 0.0 else 0.0

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

from matplotlib.dates import DateFormatter

from gluonts.mx import SimpleFeedForwardEstimator, Trainer, DeepVAREstimator
from gluonts.mx.distribution import MultivariateGaussianOutput
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.artificial import default_synthetic
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation import make_evaluation_predictions
from gluonts.lab import plot


def univariate_example():
    dataset = get_dataset("m4_hourly")

    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=dataset.metadata.prediction_length,
        context_length=100,
        trainer=Trainer(
            ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100
        ),
    )

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
        num_samples=100,
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    ts_entry = tss[0]
    forecast_entry = forecasts[0]

    fig, ax = plot(
        forecast=forecast_entry,
        timeseries=ts_entry[-100 : -dataset.metadata.prediction_length + 1],
        label_prefix="first entry",
        show_mean=True,
        train_test_separator=forecast_entry.index[0].to_timestamp(),
        show_plot=False,
    )

    # do more custom things before showing the plot
    ax.tick_params(axis="x", labelrotation=45)
    ax.xaxis.set_major_formatter(DateFormatter("%d-%m-%Y %H:%M"))
    ax.legend(loc="upper left")
    fig.show()


def multivariate_example():
    def load_multivariate_synthetic_dataset():
        dataset_info, train_ds, test_ds = default_synthetic()
        grouper_train = MultivariateGrouper(max_target_dim=10)
        grouper_test = MultivariateGrouper(num_test_dates=1, max_target_dim=10)
        metadata = dataset_info.metadata
        metadata.prediction_length = dataset_info.prediction_length
        return TrainDatasets(
            metadata=dataset_info.metadata,
            train=grouper_train(train_ds),
            test=grouper_test(test_ds),
        )

    dataset = load_multivariate_synthetic_dataset()
    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    metadata = dataset.metadata

    estimator = DeepVAREstimator(
        num_cells=20,
        num_layers=1,
        pick_incomplete=True,
        prediction_length=metadata.prediction_length,
        target_dim=target_dim,
        freq=metadata.freq,
        distr_output=MultivariateGaussianOutput(dim=target_dim),
        scaling=False,
        trainer=Trainer(
            epochs=1, learning_rate=1e-10, num_batches_per_epoch=1
        ),
    )

    predictor = estimator.train(training_data=dataset.train)

    forecast_it, tss = make_evaluation_predictions(dataset.test, predictor)
    forecast_entry = next(iter(forecast_it))

    fig, ax = plot(
        forecast=forecast_entry,
        timeseries=list(tss)[0],
        color=["g", "r"] + ["_"] * 8,
        variates_to_plot=[0, 1],
        show_plot=False,
    )

    # do more custom things before showing the plot
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="upper left")
    fig.show()


univariate_example()
multivariate_example()

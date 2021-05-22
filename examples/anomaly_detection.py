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

"""
This example shows how to do anomaly detection with DeepAR.
The model is first trained and then time-points with the largest negative log-likelihood are plotted.
"""
import numpy as np
from itertools import islice
import mxnet as mx
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from gluonts.dataset.loader import TrainDataLoader
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.mx.trainer import Trainer
from gluonts.dataset.repository.datasets import get_dataset


if __name__ == "__main__":

    dataset = get_dataset(dataset_name="electricity")

    estimator = DeepAREstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(
            learning_rate=1e-3, epochs=50, num_batches_per_epoch=100
        ),
    )

    # instead of calling `train` method, we call `train_model` that returns more things including the training model
    train_output = estimator.train_model(dataset.train)

    # we construct a data_entry that contains 500 random windows
    batch_size = 500
    num_samples = 100
    training_data_loader = TrainDataLoader(
        dataset=dataset.train,
        transform=train_output.transformation,
        batch_size=batch_size,
        num_batches_per_epoch=estimator.trainer.num_batches_per_epoch,
        ctx=mx.cpu(),
    )

    for data_entry in islice(training_data_loader, 1):
        pass

    # we now call the train model to get the predicted distribution on each window
    # this allows us to investigate where are the biggest anomalies
    context_length = train_output.trained_net.context_length
    prediction_length = train_output.trained_net.prediction_length

    input_names = get_hybrid_forward_input_names(train_output.trained_net)

    distr = train_output.trained_net.distribution(
        *[data_entry[k] for k in input_names]
    )

    # gets all information into numpy array for further plotting
    samples = distr.sample(num_samples).asnumpy()
    percentiles = np.percentile(samples, axis=0, q=[10.0, 90.0])
    target = mx.ndarray.concat(
        data_entry["past_target"], data_entry["future_target"], dim=1
    )
    target = target[:, -(context_length + prediction_length) :]
    nll = -distr.log_prob(target).asnumpy()
    target = target.asnumpy()
    mean = samples.mean(axis=0)
    percentiles = np.percentile(samples, axis=0, q=[10.0, 90.0])

    # NLL indices from largest to smallest
    sorted_indices = np.argsort(nll.reshape(-1))[::-1]

    # shows the series and times when the 20 largest NLL were observed
    for k in sorted_indices[:20]:
        i = k // nll.shape[1]
        t = k % nll.shape[1]

        time_index = pd.date_range(
            pd.Timestamp(data_entry["forecast_start"][i]),
            periods=context_length + prediction_length,
        )
        time_index -= context_length * time_index.freq

        plt.figure(figsize=(10, 4))
        plt.fill_between(
            time_index,
            percentiles[0, i],
            percentiles[-1, i],
            alpha=0.5,
            label="80% CI predicted",
        )
        plt.plot(time_index, target[i], label="target")
        plt.axvline(time_index[t], alpha=0.5, color="r")
        plt.title(f"NLL: {nll[i, t]}")
        plt.legend()
        plt.show()

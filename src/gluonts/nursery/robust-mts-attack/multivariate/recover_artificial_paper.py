# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import pandas as pd
import mxnet as mx

# First-party imports
from gluonts.distribution import (
    LowrankMultivariateGaussianOutput,
)
from gluonts.dataset.loader import TrainDataLoader
from gluonts.multivariate.datasets.sinus_covariance import sinus_covariance
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.trainer import Trainer


def recover_artificial_paper(dim=4, rank=2, epochs=30):
    print("recover artificial example from paper")
    ds, Sigma_true = sinus_covariance(max_target_dim=dim, rank=rank)

    distr_output = LowrankMultivariateGaussianOutput(
        rank=rank, dim=dim, sigma_init=1.0
    )
    context_length = 2 * ds.prediction_length

    for epochs in [epochs]:

        estimator = DeepVAREstimator(
            distr_output=distr_output,
            num_cells=40,
            num_layers=2,
            prediction_length=ds.prediction_length,
            context_length=context_length,
            target_dim=ds.target_dim,
            freq=ds.freq,
            scaling=False,
            lags_seq=[1, 24, 168],
            trainer=Trainer(
                epochs=epochs,
                init="Uniform",
                learning_rate=1e-2,
                patience=4,
                num_batches_per_epoch=100,
                hybridize=True,
                batch_size=32,
            ),
            pick_incomplete=False,
        )

        train_output = estimator.train_model(ds.train_ds)

        # todo adapt loader to anomaly detection use-case
        training_data_loader = TrainDataLoader(
            dataset=ds.test_ds,
            transform=train_output.transformation,
            batch_size=1,
            num_batches_per_epoch=estimator.trainer.num_batches_per_epoch,
            ctx=mx.cpu(),
        )

        for data_entry in islice(training_data_loader, 1):
            input_names = get_hybrid_forward_input_names(
                train_output.trained_net
            )

            loss, likelihoods, *distr_args = train_output.trained_net(
                *[data_entry[k] for k in input_names]
            )

            # (batch, seq_len, target_dim, target_dim)
            distr = distr_output.distr_cls(rank, *distr_args, ds.target_dim)

            # mu = distr.mean.asnumpy()
            # plt.plot(mu[0, :], label='mu', color='black')

            Sigma = distr.variance.asnumpy()

            cm = plt.get_cmap('Accent_r')

            start = min([pd.Timestamp(x['start']) for x in ds.train_ds])

            fcst_index = int(
                (data_entry['forecast_start'][0] - start).total_seconds()
                / 3600
            )

            for i in range(0, ds.target_dim):
                for j in range(i, ds.target_dim):

                    color = cm.colors[(i * ds.target_dim + j) % len(cm.colors)]
                    plt.plot(
                        Sigma[0, : 24 * 3, i, j],
                        color=color,
                        ls='--',
                        alpha=0.5,
                    )

                    plt.plot(
                        Sigma_true[
                            fcst_index
                            - context_length : fcst_index
                            + ds.prediction_length,
                            i,
                            j,
                        ],
                        label=f"sigma_true_{i+1}{j+1}",
                        color=color,
                    )
            plt.tight_layout()
            plt.legend()
            plt.show()

            Sigma_true_slice = Sigma_true[
                fcst_index
                - context_length : fcst_index
                + ds.prediction_length,
                :,
                :,
            ]

            diff = np.abs(Sigma_true_slice - Sigma[0, : 24 * 3, :, :])
            return (diff / np.maximum(np.abs(Sigma_true_slice), 0.001)).mean()


if __name__ == '__main__':
    mae = recover_artificial_paper(dim=4, rank=2, epochs=20)
    print(f"mae: {mae}")

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

# If you use this code in your work please cite:
# Multivariate Time Series Forecasting with Latent Graph Inference
# (https://arxiv.org/abs/2203.03423)


from typing import Type, Union
import torch
import logging
import numpy as np
from gluonts.evaluation import MultivariateEvaluator
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets
from gluonts.torch.model.gnninf import GNNInfEstimator
from gluonts.torch.model.gnninf.networks.model import GNNInfNetwork
from gluonts.evaluation import make_evaluation_predictions

logger = logging.getLogger(__name__)


class GNNInfExperiment:
    def __init__(
        self,
        dataset: TrainDatasets,
        estimator: Union[Type[Estimator], Type[Predictor]],
        batch_size: int = 4,
        num_batches_per_epoch: int = 100,
        epochs: int = 1,
        lr: float = 2e-4,
        gnn_name: str = "gnn",
        nf: int = 64,
        gnn_layers: int = 2,
        device="cpu",
    ) -> None:
        self.dataset = dataset
        self.estimator = estimator
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.epochs = epochs
        self.lr = lr
        self.gnn_name = gnn_name  # "gnn" or "bpgnn"
        self.nf = nf
        self.gnn_layers = gnn_layers
        self.device = device
        self.target_dim = int(
            self.dataset.metadata.feat_static_cat[0].cardinality
        )

    def run(self) -> None:
        metadata, train_dataset, test_dataset = (
            self.dataset.metadata,
            self.dataset.train,
            self.dataset.test,
        )
        freq = metadata.freq
        prediction_length = metadata.prediction_length
        context_length = 10 * prediction_length

        # Retrieve estimator
        estimator = self.estimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
            lr=self.lr,
            gnn_name=self.gnn_name,
            nf=self.nf,
            gnn_layers=self.gnn_layers,
            device=self.device,
        )
        # Train
        predictor = estimator.train(train_dataset)

        # Evaluate Model
        evaluator = MultivariateEvaluator(quantiles=(np.arange(10) / 10)[1:])
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset, predictor=predictor
        )
        forecasts_pytorch = list(f.to_sample_forecast() for f in forecast_it)
        tss_pytorch = list(ts_it)
        agg_metrics, item_metrics = evaluator(
            iter(tss_pytorch),
            iter(forecasts_pytorch),
            num_series=len(test_dataset),
        )

        for name in ["mean_wQuantileLoss", "RMSE", "ND"]:
            print(f"{name}", agg_metrics[name])


def retrieve_multivariate_dataset(
    dataset_name: str, regenerate: bool = False, max_num_nodes: int = 100000
) -> TrainDatasets:
    dataset = get_dataset(dataset_name, regenerate=regenerate)
    len_test, len_train = len(dataset.test), len(dataset.train)
    num_test_dates = float(len_test) / float(len_train)
    assert num_test_dates == int(num_test_dates)
    grouper_train = MultivariateGrouper(max_target_dim=max_num_nodes)
    grouper_test = MultivariateGrouper(
        max_target_dim=max_num_nodes, num_test_dates=int(num_test_dates)
    )
    return TrainDatasets(
        metadata=dataset.metadata,
        train=grouper_train(dataset.train),
        test=grouper_test(dataset.test),
    )


def test_gluonts_gnninf() -> None:
    dataset = retrieve_multivariate_dataset("exchange_rate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expt = GNNInfExperiment(
        dataset=dataset, estimator=GNNInfEstimator, device=device
    )
    expt.run()


def test_torch_gnninf() -> None:
    # Parameters init
    in_channels = 1
    out_channels = 1
    input_length = 6
    pred_length = 3
    num_nodes = 10
    bs = 16
    agg_name = "gnn"  # gnn or bpgnn

    # Model init
    fcgnn = GNNInfNetwork(
        in_channels,
        out_channels,
        input_length,
        pred_length,
        num_nodes,
        agg_name=agg_name,
    )

    # Data init
    batch = torch.zeros(bs, num_nodes, input_length, in_channels)

    # Model forward
    out = fcgnn(batch)

    # Print output size
    print("Output size %s" % str(out.size()))

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

import argparse

import torch
import torch.nn.functional as F

from gluonts.torch import DeepAREstimator
from gluonts.dataset.repository import get_dataset
from gluonts.itertools import select
from gluonts.torch.util import take_last, lagged_sequence_values
from gluonts.torch.distributions import GeneralizedPareto
from gluonts.dataset.field_names import FieldName


def fit_gpd(data, num_iterations=100, learning_rate=0.001):
    """
    Fit a Generalized Pareto Distribution to the given data using RMSprop optimizer.

    Args:
    data (torch.Tensor): Input tensor of shape (batch_size, num_samples)
    num_iterations (int): Number of optimization iterations
    learning_rate (float): Learning rate for the optimizer

    Returns:
    tuple: Fitted parameters (loc, scale, concentration)
    """
    batch_size, _ = data.shape

    # Initialize parameters
    loc = data.min(dim=1, keepdim=True)[0] - 1
    loc.requires_grad = True
    scale = torch.ones(batch_size, 1, device=data.device, requires_grad=True)
    concentration = torch.ones(
        batch_size, 1, device=data.device, requires_grad=True
    ).div_(3)

    optimizer = torch.optim.RMSprop(
        [loc, scale, concentration], lr=learning_rate
    )
    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    def _gdk_domain_map(loc, scale, concentration):
        scale = F.softplus(scale)
        neg_conc = concentration < 0
        loc = torch.where(neg_conc, loc - scale / concentration, loc)
        return GeneralizedPareto(loc, scale, concentration)

    def closure():
        optimizer.zero_grad()
        gpd = _gdk_domain_map(loc, scale, concentration)
        loss = -gpd.log_prob(data).mean()
        print(f"Loss: {loss.item()}")
        loss.backward()
        lr_scheduler.step(loss)
        return loss

    for _ in range(num_iterations):
        optimizer.step(closure)

    return GeneralizedPareto(
        loc.detach(), scale.detach(), concentration.detach()
    )


def main(args):
    dataset = get_dataset(dataset_name=args.dataset)
    estimator = DeepAREstimator(
        prediction_length=dataset.metadata.prediction_length,
        context_length=args.context_length,
        freq=dataset.metadata.freq,
        trainer_kwargs=dict(
            max_epochs=args.max_epochs,
        ),
        batch_size=args.batch_size,
    )
    predictor = estimator.train(dataset.train, cache_data=True)
    print(f"Training completed for dataset: {args.dataset}")
    model = predictor.prediction_net.model

    # Load the test dataset
    transformation = estimator.create_transformation()
    transformed_test_data = transformation.apply(dataset.test, is_train=True)
    test_data_loader = estimator.create_validation_data_loader(
        transformed_test_data,
        predictor.prediction_net,
    )

    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            inputs = select(
                predictor.input_names + [f"future_{FieldName.TARGET}"],
                batch,
                ignore_missing=True,
            )
            params, scale, _, static_feat, state = model.unroll_lagged_rnn(
                inputs["feat_static_cat"],
                inputs["feat_static_real"],
                inputs["past_time_feat"],
                inputs["past_target"],
                inputs["past_observed_values"],
                inputs["future_time_feat"][:, :1],
            )
            # remove the very last param from the params
            sliced_params = [p[:, :-1] for p in params]
            distr = model.output_distribution(sliced_params)

            # get the last target and calcualte its anomaly score
            context_target = take_last(
                inputs["past_target"], dim=-1, num=model.context_length - 1
            )
            # calculate the surprisal scores for the context target
            scores = -distr.log_prob(context_target / scale)

            # get the top 10% of the scores for each time series of the batch
            top_scores = torch.topk(
                scores, k=int(scores.shape[1] * 0.1), dim=1
            )
            # get top scores [B, 10% of context_length]
            top_scores = top_scores.values

            # fit a Generalized Pareto Distribution to the top_scores aka surprisal scores
            gpd = fit_gpd(top_scores)

            # Loop over each prediction length
            scaled_future_target = inputs["future_target"] / scale
            distr = model.output_distribution(params, trailing_n=1)
            anomalies = []
            for i in range(scaled_future_target.shape[1]):
                score = -distr.log_prob(scaled_future_target[:, i : i + 1])
                # check if the score are less than gpd.loc? for each entry in the batch
                is_anomaly = score < gpd.loc
                # mask out the score where is_anomaly is True
                score = torch.where(is_anomaly, gpd.loc + 1, score)
                is_anomaly = torch.where(
                    is_anomaly, False, gpd.cdf(score) < 0.05
                )
                anomalies.append(is_anomaly)

                next_features = torch.cat(
                    (
                        static_feat.unsqueeze(dim=1),
                        inputs["future_time_feat"][:, i : i + 1],
                    ),
                    dim=-1,
                )

                next_lags = lagged_sequence_values(
                    model.lags_seq,
                    inputs["past_target"] / scale,
                    scaled_future_target[:, i : i + 1],
                    dim=-1,
                )
                rnn_input = torch.cat((next_lags, next_features), dim=-1)
                output, state = model.rnn(rnn_input, state)
                params = model.param_proj(output)
                distr = model.output_distribution(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anomaly detection using PyTorch and GluonTS"
    )
    parser.add_argument(
        "--dataset", type=str, default="electricity", help="Dataset name"
    )
    parser.add_argument(
        "--context_length", type=int, default=None, help="Context length"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=3, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )

    args = parser.parse_args()

    if args.context_length is None:
        args.context_length = (
            get_dataset(args.dataset).metadata.prediction_length * 4
        )

    main(args)

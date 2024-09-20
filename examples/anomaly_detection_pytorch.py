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

from tqdm import tqdm
import pandas as pd

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
        GeneralizedPareto: Fitted GPD(loc, scale, concentration) distribution without any validation
    """
    batch_size, _ = data.shape

    # Initialize parameters for the GPD so that the loc is always less than the data to begin with
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

    def _gdk_domain_map(loc, scale, concentration, validate_args=None):
        scale = F.softplus(scale)
        neg_conc = concentration < 0
        loc = torch.where(neg_conc, loc - scale / concentration, loc)
        return GeneralizedPareto(
            loc, scale, concentration, validate_args=validate_args
        )

    def closure():
        optimizer.zero_grad()
        gpd = _gdk_domain_map(loc, scale, concentration)
        loss = -gpd.log_prob(data).mean()
        loss.backward()
        lr_scheduler.step(loss)
        return loss

    for _ in range(num_iterations):
        optimizer.step(closure)

    return _gdk_domain_map(
        loc.detach(),
        scale.detach(),
        concentration.detach(),
        validate_args=False,
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

    anomalies = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Processing batches"):
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

            # get the args.top_score_percentage of the scores for each time series of the batch
            top_scores = torch.topk(
                scores,
                k=int(scores.shape[1] * args.top_score_percentage),
                dim=1,
            )
            # fit a Generalized Pareto Distribution to the top_scores aka surprisal scores values
            gpd = fit_gpd(
                top_scores.values,
                num_iterations=args.gpd_iterations,
                learning_rate=args.gpd_learning_rate,
            )

            # Loop over each prediction length
            scaled_future_target = inputs["future_target"] / scale
            distr = model.output_distribution(params, trailing_n=1)
            batch_anomalies = []
            for i in tqdm(
                range(scaled_future_target.shape[1]),
                desc="Processing prediction length",
                leave=False,
            ):
                score = -distr.log_prob(scaled_future_target[:, i : i + 1])
                # only check if its an anomaly for scores greater than gpd.loc for each entry in the batch
                is_anomaly = torch.where(
                    score < gpd.loc,
                    False,
                    gpd.cdf(score) < args.anomaly_threshold,
                )
                batch_anomalies.append(is_anomaly)

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
            # stack the batch_anomalies along the prediction length dimension
            anomalies.append(torch.stack(batch_anomalies, dim=1))
        anomalies = torch.cat(anomalies, dim=0).cpu().numpy()

    # save as csv
    all_dates = []
    all_flags = []
    all_targets = []
    for i, (entry, flags) in enumerate(zip(dataset.test, anomalies)):
        start_date = entry["start"].to_timestamp()
        target = entry["target"]
        dates = pd.date_range(
            start=start_date, periods=len(target), freq=dataset.metadata.freq
        )
        # take the last prediction_length dates
        date_index = dates[-dataset.metadata.prediction_length :]
        target_slice = target[-dataset.metadata.prediction_length :]
        all_dates.append(date_index)
        all_flags.append(flags.flatten().astype(bool))
        all_targets.append(target_slice)

    # create a dataframe with the date_index and the flags
    anomaly_df = pd.DataFrame(
        {"date": all_dates, "is_anomaly": all_flags, "target": all_targets}
    )
    anomaly_df.set_index("date", inplace=True)
    anomaly_df.to_csv(f"anomalies_{args.dataset}.csv")


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
        "--max_epochs", type=int, default=30, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=0.05,
        help="Threshold for anomaly detection",
    )
    parser.add_argument(
        "--top_score_percentage",
        type=float,
        default=0.1,
        help="Percentage of top scores to consider for GPD fitting",
    )
    parser.add_argument(
        "--gpd_iterations",
        type=int,
        default=100,
        help="Number of iterations for GPD fitting",
    )
    parser.add_argument(
        "--gpd_learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for GPD fitting",
    )
    args = parser.parse_args()

    if args.context_length is None:
        args.context_length = (
            get_dataset(args.dataset).metadata.prediction_length * 10
        )

    main(args)

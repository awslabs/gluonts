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
import pickle
import torch
from tqdm.auto import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

from datasets import DATASETS, get_dataset
from utils import (
    Metrics,
    Params,
    calc_loss,
    load_pickle,
    change_device,
    smoothed_inference,
    PREDICTION_INPUT_NAMES,
)
from gluonts.model.predictor import Predictor
from pathlib import Path
from read_pickle import create_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", type=str, help="path to model checkpoint"
    )
    parser.add_argument(
        "--attack_result", type=str, help="path to the attack result"
    )
    parser.add_argument(
        "--dataset", type=str, default="electricity", choices=DATASETS
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda w/ number specified",
    )
    parser.add_argument("--tol", type=float, default=3.0)
    parser.add_argument(
        "--attack_params_path",
        type=str,
        required=True,
        help="path to json file containing attack parameters",
    )
    parser.add_argument(
        "--rs", action="store_true", help="whether to use randomized smoothing"
    )

    args = parser.parse_args()

    device = args.device
    path = Path(args.model_path)
    predictor = Predictor.deserialize(path)
    params = Params(json_path=args.attack_params_path)

    filename = args.attack_result
    attack_data = load_pickle("./attack_results_var/" + filename)
    attack_data = list(attack_data)

    ds = get_dataset(args.dataset, predictor.prediction_net.target_dim)

    max_norm = params.max_norm
    attack_idx = params.attack_idx
    target_items = params.target_items
    attack_items = params.attack_items
    sparsity = params.sparsity

    net = predictor.prediction_net.to(device)
    if args.rs:
        net.num_parallel_samples = 1
    net.eval()
    context_length = net.context_length
    prediction_length = net.prediction_length
    target_dim = net.target_dim

    forecasts_keys = ["no attack"] + [str(k) for k in sparsity]
    forecasts = {key: [] for key in forecasts_keys}

    for i in tqdm(range(len(attack_data))):
        batch = attack_data[i].batch
        batch = change_device(batch, device)
        tol = batch["past_target_cdf"].abs().max() * args.tol
        ground_truth = torch.Tensor(attack_data[i].true_future_target).to(
            device
        )
        with torch.no_grad():
            """
            Clean data
            """
            inputs = dict(
                [(key, batch[key]) for key in PREDICTION_INPUT_NAMES]
            )
            if args.rs:
                outputs = smoothed_inference(
                    batch, batch["past_target_cdf"], net, params.sigma, device
                )
            else:
                outputs = net(**inputs).detach().cpu().numpy()
            forecasts["no attack"].append(outputs)
            del inputs, outputs
            torch.cuda.empty_cache()
            """
            Sparse attacked data
            """
            perturbation = attack_data[i].perturbation["dense"]
            batch_size = perturbation.shape[0]
            for k in sparsity:
                perturbation_tensor = change_device(perturbation, device)
                for idx in range(perturbation_tensor.shape[-1]):
                    if idx not in params.attack_items:
                        perturbation_tensor[:, :, idx] = 0
                top_k_idx = (
                    perturbation_tensor.abs()
                    .sum(1)
                    .argsort(dim=-1, descending=True)
                )
                for each_batch in range(batch_size):
                    for idx in range(perturbation_tensor.shape[-1]):
                        if idx not in top_k_idx[each_batch][:k]:
                            perturbation_tensor[each_batch, :, idx] = 0
                attacked_past_target = change_device(
                    batch["past_target_cdf"], device
                ) * (1 + perturbation_tensor)
                attacked_inputs = dict(
                    [(key, batch[key]) for key in PREDICTION_INPUT_NAMES]
                )
                attacked_inputs["past_target_cdf"] = attacked_past_target

                if args.rs:
                    outputs = smoothed_inference(
                        batch, attacked_past_target, net, params.sigma, device
                    )
                else:
                    outputs = net(**attacked_inputs).detach().cpu().numpy()
                forecasts[str(k)].append(outputs)

                del perturbation_tensor, attacked_inputs, outputs
                torch.cuda.empty_cache()

    with torch.no_grad():
        mse, mape, ql = calc_loss(
            attack_data,
            forecasts,
            attack_idx=attack_idx,
            target_items=target_items,
        )
        for key in ql.keys():
            ql[key] = ql[key].mean(0)
        metrics = Metrics(mse=mse, mape=mape, ql=ql)

    if not args.rs:
        if not os.path.exists("./metrics"):
            os.makedirs("./metrics")
        with open("./metrics/" + filename, "wb") as outp:
            pickle.dump(metrics, outp, pickle.HIGHEST_PROTOCOL)
            pickle_path = "./metrics/" + filename
    else:
        if not os.path.exists("./metrics_rs"):
            os.makedirs("./metrics_rs")
        with open("./metrics_rs/" + filename, "wb") as outp:
            pickle.dump(metrics, outp, pickle.HIGHEST_PROTOCOL)
            pickle_path = "./metrics_rs/" + filename

    print(pickle_path)
    create_table(pickle_path)

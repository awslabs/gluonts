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
import warnings

warnings.filterwarnings("ignore")
import pickle
import numpy as np
import os
from attack_modules.attack_var import SparseLayerAttack
from datasets import DATASETS, get_dataset
from utils import (
    AttackResults,
    Params,
    PREDICTION_INPUT_NAMES,
    change_device,
    ts_iter,
    requires_grad_,
)
from gluonts.model.predictor import Predictor
from pathlib import Path
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.torch.batchify import batchify

parser = argparse.ArgumentParser()

parser.add_argument("model_path", type=str, help="path to the saved model")
parser.add_argument(
    "--dataset", type=str, default="electricity", choices=DATASETS
)
parser.add_argument(
    "--attack_params", type=str, help="path to attack parameters"
)

args = parser.parse_args()


def main():
    # load predictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "gaussian" in args.model_path and "adv" in args.model_path:
        model_type = "adv_rt"
        sparsity = args.model_path[-2]
    elif "adv" in args.model_path:
        model_type = "adv"
        sparsity = args.model_path[-2]
    elif "gaussian" in args.model_path:
        model_type = "rt"
    else:
        model_type = "vanilla"

    path = Path(args.model_path)
    predictor = Predictor.deserialize(path)
    predictor.prediction_net = predictor.prediction_net.to(device)

    # load parameters
    prediction_length = predictor.prediction_length
    target_dim = predictor.prediction_net.target_dim
    ds = get_dataset(args.dataset, target_dim)

    test_loader = InferenceDataLoader(
        ds.test_ds,
        transform=predictor.input_transform,
        batch_size=predictor.batch_size,
        stack_fn=lambda data: batchify(data, predictor.device),
    )
    requires_grad_(predictor.prediction_net, False)
    predictor.prediction_net.eval()

    tss = list(ts_iter(ds.test_ds, ds.freq))
    true_future_targets = [
        tss[i].to_numpy()[-prediction_length:] for i in range(len(tss))
    ]
    testset_idx = 0

    params = Params(json_path=args.attack_params)
    params.device = device
    attack_params_id = args.attack_params[-6]

    attack = SparseLayerAttack(
        model=predictor.prediction_net,
        params=params,
        input_names=PREDICTION_INPUT_NAMES,
    )

    filename = (
        "./attack_results_sparse/"
        + model_type
        + "_"
        + str(prediction_length)
        + "_"
        + args.dataset
        + "_"
        + "rank"
        + str(predictor.prediction_net.distr_output.rank)
        + "_dim"
        + str(predictor.prediction_net.distr_output.dim)
    )

    if not os.path.exists("./attack_results_sparse"):
        os.makedirs("./attack_results_sparse")

    if "adv" in model_type:
        filename += "_s" + str(sparsity)
    filename += "_" + attack_params_id + ".pkl"

    with open(filename, "wb") as outp:
        for batch in test_loader:
            batch = change_device(batch, params.device)
            target = batch["past_target_cdf"]
            this_batch = target.shape[0]

            future_target = np.array(
                true_future_targets[testset_idx : testset_idx + this_batch]
            )
            best_perturbation = attack.attack_batch(
                batch,
                true_future_target=future_target
                if device == "cpu"
                else torch.from_numpy(future_target).float().to(device),
            )

            batch_res = AttackResults(
                batch=batch,
                perturbation=best_perturbation,
                true_future_target=future_target,
                tolerance=params.tolerance,
                attack_idx=params.attack_idx,
            )

            pickle.dump(batch_res, outp, pickle.HIGHEST_PROTOCOL)
            testset_idx += this_batch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

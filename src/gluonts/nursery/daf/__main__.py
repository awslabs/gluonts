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


from pathlib import Path
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch as pt

from .dataset import (
    FourierDataset,
    BenchmarkDataset,
    DomAdaptDataset,
)
from .estimator import (
    AttentionEstimator,
    AdversarialEstimator,
    DomAdaptEstimator,
    AdversarialDomAdaptEstimator,
)
from .engine import (
    AttentionTrainer,
    AttentionEvaluator,
    DomAdaptTrainer,
    DomAdaptEvaluator,
    AdversarialDomAdaptTrainer,
    AdversarialDomAdaptEvaluator,
)
from .cli import (
    NestedArgumentParser,
    NestedNamespace,
)


_freqs = {
    "electricity": "H",
    "traffic": "H",
    "wiki10k": "D",
    "kaggle": "D",
}

_window_size = {
    "H": 24 * 8,
    "D": 7 * 5,
}

_horizons = {
    "H": 24,
    "D": 7,
}


def update_configs(
    configs: NestedNamespace,
    defaults: dict,
) -> NestedNamespace:
    for key, value in defaults.items():
        if isinstance(value, dict):
            if isinstance(getattr(configs, key), NestedNamespace):
                namespace = getattr(configs, key)
                update_configs(namespace, value)
            else:
                raise ValueError("Incompatible structures.")
        else:
            if isinstance(getattr(configs, key), NestedNamespace):
                raise ValueError("Incompatible structures.")
            elif getattr(configs, key) is None:
                setattr(configs, key, value)


def parse_configs(cmd=None):
    parser = NestedArgumentParser()

    data_configs = parser.add_argument_group("data")
    data_configs.add_argument("src_name", type=str)
    data_configs.add_argument("tgt_name", type=str)
    data_configs.add_argument("--n_src_instances_train", type=int)
    data_configs.add_argument("--n_tgt_instances_train", type=int)
    data_configs.add_argument("--n_src_instances_eval", type=int)
    data_configs.add_argument("--n_tgt_instances_eval", type=int)
    data_configs.add_argument("--src_eval_size", type=int)
    data_configs.add_argument("--tgt_eval_size", type=int)

    model_configs = parser.add_argument_group("model")
    model_configs.add_argument("--d_hidden", type=int)
    model_configs.add_argument("--n_layer", type=int)
    model_configs.add_argument("--window_size", nargs="+", type=int)
    model_configs.add_argument("--n_head", type=int)
    model_configs.add_argument("--n_enc_layer", type=int)
    model_configs.add_argument("--n_dec_layer", type=int)
    model_configs.add_argument("--n_disc_layer", type=int)
    model_configs.add_argument("--symmetric", action="store_true")
    model_configs.add_argument("--share_values", action="store_true")
    model_configs.add_argument("--tie_casts", action="store_true")
    model_configs.add_argument("--tie_layers", action="store_true")
    model_configs.add_argument("--dropout", type=float)
    model_configs.add_argument("--temperature", type=float)
    model_configs.add_argument(
        "--not_scale_input", action="store_false", dest="normalize_input"
    )
    model_configs.add_argument(
        "--not_scale_output", action="store_false", dest="rescale_output"
    )
    model_configs.add_argument(
        "--last_layer_loss_only", action="store_false", dest="layerwise_loss"
    )

    train_configs = parser.add_argument_group("train")
    train_configs.add_argument("--lr", type=float)
    train_configs.add_argument("--n_epochs", type=int)
    train_configs.add_argument("--nb_epoch", type=int)
    train_configs.add_argument("--batch_size", type=int, nargs="+")
    train_configs.add_argument("--max_grad_norm", type=float)
    train_configs.add_argument("--n_loader_workers", type=int)
    train_configs.add_argument("--betas", type=float, nargs="+")
    train_configs.add_argument("--weight_decay", type=float)
    train_configs.add_argument("--cuda_device", type=int)

    global_configs = parser.add_argument_group("global")
    global_configs.add_argument("--seed", type=int)

    args = parser.parse_args(cmd)
    return args


def get_dataset(
    src_name: str,
    tgt_name: str,
    n_src_instances_train: int,
    n_tgt_instances_train: int,
    n_src_instances_eval: int,
    n_tgt_instances_eval: int,
    src_eval_size: int,
    tgt_eval_size: int,
):
    src_dataset = BenchmarkDataset.create_dataset(
        dataset=src_name,
        freq=_freqs[src_name],
        n_instances_train=n_src_instances_train,
        n_instances_eval=n_src_instances_eval,
        eval_size=src_eval_size,
        window_size=_window_size[_freqs[src_name]],
    )
    tgt_dataset = BenchmarkDataset.create_dataset(
        dataset=tgt_name,
        freq=_freqs[tgt_name],
        n_instances_train=n_tgt_instances_train,
        n_instances_eval=n_tgt_instances_eval,
        eval_size=tgt_eval_size,
        train_size=tgt_eval_size,
        window_size=_window_size[_freqs[tgt_name]],
    )
    return src_dataset, tgt_dataset


def main():
    args = parse_configs()
    src_name = args.data.src_name
    tgt_name = args.data.tgt_name
    exp_name = f"{src_name[0].upper()}2{tgt_name[0].upper()}"
    with Path(__file__).parents[1].joinpath(
        f"dumps/{exp_name}/defaults.json"
    ).open("r") as f:
        defaults = json.load(f)
    update_configs(args, defaults)
    src_dataset, tgt_dataset = get_dataset(**vars(args.data))

    n_disc_layer = args.model.n_disc_layer
    delattr(args.model, "n_disc_layer")
    seed = getattr(args, "global").seed

    pt.manual_seed(seed)
    S1 = AttentionEstimator.from_configs(
        src_dataset.d_data,
        src_dataset.d_feats,
        horizon=_horizons[_freqs[src_name]],
        **vars(args.model),
    )
    args.train.log_dir = f"dumps/{exp_name}/src"
    S1_trainer = AttentionTrainer.from_configs(
        src_dataset,
        S1,
        **vars(args.train),
    )
    pt.cuda.manual_seed_all(seed)
    S1_trainer.fit()
    S1_evaluator = AttentionEvaluator.from_trainer(S1_trainer, "best")
    S1_evaluator.evaluate()
    print(S1_evaluator.metrics.test.value)

    pt.manual_seed(seed)
    T0 = AttentionEstimator.from_configs(
        tgt_dataset.d_data,
        tgt_dataset.d_feats,
        horizon=_horizons[_freqs[tgt_name]],
        **vars(args.model),
    )
    args.train.log_dir = f"dumps/{exp_name}/tgt"
    T0_trainer = AttentionTrainer.from_configs(
        tgt_dataset,
        T0,
        **vars(args.train),
    )
    pt.cuda.manual_seed_all(seed)
    T0_trainer.fit()
    T0_evaluator = AttentionEvaluator.from_trainer(T0_trainer, "best")
    T0_evaluator.evaluate()
    print(T0_evaluator.metrics.test.value)

    DA_dataset = DomAdaptDataset.from_domains(src_dataset, tgt_dataset)
    T1 = S1.create_twin_estimator(
        tgt_dataset.d_data,
        tgt_dataset.d_feats,
        horizon=_horizons[_freqs[tgt_name]],
    )
    DA = DomAdaptEstimator(S1, T1)
    args.train.log_dir = f"dumps/{exp_name}/da"
    DA_trainer = DomAdaptTrainer.from_configs(
        DA_dataset,
        DA,
        **vars(args.train),
    )
    DA_trainer.fit()
    DA_evaluator = DomAdaptEvaluator.from_trainer(DA_trainer, "best")
    DA_evaluator.evaluate()
    print(DA_evaluator.metrics.test.value)

    S1_evaluator = AttentionEvaluator.from_trainer(S1_trainer, "best")
    S2 = AdversarialEstimator.from_base(S1, n_disc_layer)
    T2 = S2.create_twin_estimator(
        tgt_dataset.d_data,
        tgt_dataset.d_feats,
        horizon=_horizons[_freqs[tgt_name]],
    )
    ADV = AdversarialDomAdaptEstimator(S2, T2)
    args.train.log_dir = f"dumps/{exp_name}/adv"
    ADV_trainer = AdversarialDomAdaptTrainer.from_configs(
        DA_dataset,
        ADV,
        **vars(args.train),
    )
    ADV_trainer.fit()
    ADV_evaluator = AdversarialDomAdaptEvaluator.from_trainer(
        ADV_trainer, "best"
    )
    ADV_evaluator.evaluate()
    print(ADV_evaluator.metrics.test.value)


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
import torch

from models.switching_gaussian_linear_system import SGLSPredictor
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    create_loaders,
    get_cardinalities,
    get_dataset,
)
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.dataset.common import ListDataset
from utils.utils import prepare_logging
from gluonts.evaluation import Evaluator
import importlib.util


def load_model_config(log_paths):
    spec = importlib.util.spec_from_file_location(
        "config", os.path.join(log_paths.root, "config.py")
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


# def load_model(log_paths, config_file, epoch: (int, str, None), device):
#     model = config_file.make_model(config)
#     epoch = epoch if epoch is not None else "best"
#     state_dict = torch.load(os.path.join(log_paths.model, f"{epoch}.pt"))
#     model.load_state_dict(state_dict)
#     # try:
#     #     model.load_state_dict(state_dict)
#     # except:
#     #     model.load_state_dict({key.replace("module.", ""): val for key, val in state_dict.items()})
#     model = model.to(device=device)
#     return model


def test(
    model,
    dataset,
    log_paths,
    input_transforms,
    cardinalities,
    config,
    which="best",
):
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(torch.load(os.path.join(log_paths.model, f"{which}.pt")))
    m.n_particle = config.num_samples_eval

    forecaster_test_full = SGLSPredictor(
        model=model,
        input_transform=input_transforms["test_full"],
        batch_size=config.batch_size_val,
        prediction_length=config.prediction_length_full,
        freq=dataset.metadata.freq,
        lead_time=0,
        cardinalities=cardinalities,
        dims=config.dims,
        bias_y=config.normalisation_params[0],
        factor_y=config.normalisation_params[1],
        time_feat=config.time_feat,
        keep_filtered_predictions=False,
        yield_forecast_only=True,
    )
    forecaster_test_rolling = SGLSPredictor(
        model=model,
        input_transform=input_transforms["test_rolling"],
        batch_size=config.batch_size_val,
        prediction_length=config.prediction_length_rolling,
        freq=dataset.metadata.freq,
        lead_time=0,
        cardinalities=cardinalities,
        dims=config.dims,
        bias_y=config.normalisation_params[0],
        factor_y=config.normalisation_params[1],
        time_feat=config.time_feat,
        keep_filtered_predictions=False,
        yield_forecast_only=True,
    )

    n_roll_repetitions = int(
        config.prediction_length_full / config.prediction_length_rolling
    )
    assert n_roll_repetitions == int(len(dataset.test) / len(dataset.train))
    len_test_full = int(len(dataset.test) / n_roll_repetitions)
    dataset_test_full = ListDataset(dataset.test, freq=dataset.metadata.freq)
    dataset_test_full.list_data = dataset_test_full.list_data[-len_test_full:]
    dataset_test_rolling = ListDataset(
        dataset.test, freq=dataset.metadata.freq
    )

    with torch.no_grad():
        agg_metrics_full, item_metrics_full = backtest_metrics(
            train_dataset=None,
            test_dataset=dataset_test_full,
            forecaster=forecaster_test_full,
            num_samples=config.num_samples_eval,
            num_workers=0,
            evaluator=Evaluator(
                quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,),
                num_workers=0,
            ),
        )
        np.savez(
            os.path.join(log_paths.metrics, f"agg_metrics_full_{which}.npz"),
            agg_metrics_full,
        )
        print("results full: \n", agg_metrics_full)
        agg_metrics_rolling, item_metrics_rolling = backtest_metrics(
            train_dataset=None,
            test_dataset=dataset_test_rolling,
            forecaster=forecaster_test_rolling,
            num_samples=config.num_samples_eval,
            num_workers=0,
            evaluator=Evaluator(
                quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,),
                num_workers=0,
            ),
        )
        np.savez(
            os.path.join(
                log_paths.metrics, f"agg_metrics_rolling_{which}.npz"
            ),
            agg_metrics_rolling,
        )
        print("results rolling: \n", agg_metrics_rolling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provide path to a trained model"
    )
    parser.add_argument(
        "-experiment_root_log_path",
        type=str,
        help="path to the root folder of the experiment",
    )
    parser.add_argument(
        "-epoch",
        type=int,
        default=None,
        help="the epoch to load. default latest stored",
    )
    parser.add_argument(
        "-num_samples",
        type=int,
        default=100,
        help="number of particles in eval.",
    )
    parser.add_argument(
        "-num_workers",
        type=int,
        default=0,
        help="worker for data loader. Default 0 uses no multi-processing.",
    )
    parser.add_argument(
        "-batch_size", type=int, default=7, help="batch size for evaluation."
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cpu",
        help="consider default evaluate on CPU due to ~5/4 use of RAM.",
    )

    args = parser.parse_args()
    print(args)
    print(
        "Loading default config... "
        "If file was changed programatically, then this must still be done here too."
    )
    log_paths = prepare_logging(
        config=None,
        consts=None,
        copy_config_file=False,
        root_log_path=args.experiment_root_log_path,
    )
    dataset_name = log_paths.root.split("/")[-3]
    config_file = load_model_config(log_paths=log_paths)
    config = config_file.make_default_config(dataset_name=dataset_name)
    dims = config.dims

    dataset = get_dataset(config.dataset_name)
    (
        train_loader,
        val_loader,
        test_full_loader,
        test_rolling_loader,
        input_transforms,
    ) = create_loaders(
        dataset=dataset,
        batch_sizes={
            "train": dims.batch,
            "val": config.batch_size_val,
            "test_full": config.batch_size_val,
            "test_rolling": config.batch_size_val,
        },
        past_length=config.dims.timesteps,
        prediction_length_full=config.prediction_length_full,
        prediction_length_rolling=config.prediction_length_rolling,
        num_workers=0,
        extract_tail_chunks_for_train=config.extract_tail_chunks_for_train,
    )
    cardinalities = get_cardinalities(
        dataset=dataset, add_trend=config.add_trend
    )

    model = config_file.make_model(config=config)
    epoch = args.epoch if args.epoch is not None else "best"
    state_dict = torch.load(os.path.join(log_paths.model, f"{epoch}.pt"))
    model.load_state_dict(state_dict)
    model = model.to(device=args.device)

    test(
        model=model,
        dataset=dataset,
        log_paths=log_paths,
        input_transforms=input_transforms,
        cardinalities=cardinalities,
        config=config,
        which=epoch,
    )
    print("Done.")

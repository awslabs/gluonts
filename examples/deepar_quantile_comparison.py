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
DeepAR Comparison: NormalOutput vs QuantileOutput

Compares DeepAR with distribution-based (NormalOutput) and quantile regression
(QuantileOutput) outputs on synthetic sine wave data.

Usage:
    python examples/deepar_quantile_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple

from lightning import seed_everything

# Workaround for PyTorch 2.6+ weights_only=True default in torch.load
# which rejects arbitrary globals during checkpoint deserialization.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NormalOutput
from gluonts.torch.distributions.quantile_output import QuantileOutput


def create_sine_dataset(
    num_series: int = 20,
    length: int = 200,
    prediction_length: int = 24,
    freq: str = "H",
    noise_std: float = 0.3,
    seed: int = 42,
) -> Tuple[ListDataset, ListDataset]:
    """Create synthetic sine wave dataset with Gaussian noise."""
    rng = np.random.RandomState(seed)
    start = pd.Period("2023-01-01 00:00", freq=freq)

    train_entries = []
    test_entries = []

    for i in range(num_series):
        amplitude = 1.0 + rng.rand() * 2.0
        period = 24 + rng.randint(-4, 5)
        phase = rng.rand() * 2 * np.pi
        offset = rng.rand() * 3.0 + 1.0

        t = np.arange(length)
        values = offset + amplitude * np.sin(2 * np.pi * t / period + phase)
        values += rng.randn(length) * noise_std

        train_entries.append(
            {"start": start, "target": values[:-prediction_length]}
        )
        test_entries.append({"start": start, "target": values})

    train_ds = ListDataset(train_entries, freq=freq)
    test_ds = ListDataset(test_entries, freq=freq)
    return train_ds, test_ds


def compute_quantile_loss(
    actual: np.ndarray, predicted: np.ndarray, q: float
) -> float:
    """Compute pinball (quantile) loss."""
    errors = actual - predicted
    return float(np.mean(np.maximum(q * errors, (q - 1) * errors)))


def train_and_evaluate(
    train_ds,
    test_ds,
    distr_output,
    freq: str,
    prediction_length: int,
    num_epochs: int = 20,
    model_name: str = "Model",
) -> Tuple[dict, List, List]:
    """Train a DeepAR model and evaluate on test data."""
    print(f"\nTraining DeepAR with {model_name}...")

    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=prediction_length * 2,
        num_layers=2,
        hidden_size=40,
        dropout_rate=0.1,
        distr_output=distr_output,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={
            "max_epochs": num_epochs,
            "enable_progress_bar": True,
            "enable_model_summary": False,
        },
    )

    predictor = estimator.train(train_ds)

    forecast_it, ts_it = make_evaluation_predictions(
        test_ds, predictor=predictor, num_samples=100
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Compute metrics
    quantile_levels = [0.1, 0.5, 0.9]
    metrics = {}

    all_actuals = []
    all_medians = []
    quantile_preds = {q: [] for q in quantile_levels}

    for forecast, ts in zip(forecasts, tss):
        actual = ts.values.flatten()[-prediction_length:]
        all_actuals.append(actual)
        all_medians.append(forecast.median)
        for q in quantile_levels:
            quantile_preds[q].append(forecast.quantile(q))

    all_actuals = np.concatenate(all_actuals)
    all_medians = np.concatenate(all_medians)

    metrics["RMSE"] = float(np.sqrt(np.mean((all_actuals - all_medians) ** 2)))
    metrics["MAE"] = float(np.mean(np.abs(all_actuals - all_medians)))
    metrics["ND"] = float(
        np.sum(np.abs(all_actuals - all_medians)) / np.sum(np.abs(all_actuals))
    )

    for q in quantile_levels:
        preds = np.concatenate(quantile_preds[q])
        metrics[f"QL_{q}"] = compute_quantile_loss(all_actuals, preds, q)

    print(f"{model_name} Results:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

    return metrics, forecasts, tss


def plot_forecasts(
    forecasts_normal: List,
    forecasts_quantile: List,
    tss: List,
    prediction_length: int,
    num_plots: int = 4,
    output_file: str = "deepar_quantile_comparison.png",
):
    """Plot side-by-side forecasts comparing Normal and Quantile outputs."""
    fig, axes = plt.subplots(
        num_plots, 2, figsize=(16, 3 * num_plots), sharey="row"
    )

    for idx in range(num_plots):
        if idx >= len(tss):
            break

        ts = tss[idx]
        ts_values = ts.values.flatten()

        # Show last 72 points of history + forecast
        history_show = min(72, len(ts_values) - prediction_length)
        start_idx = len(ts_values) - prediction_length - history_show
        plot_values = ts_values[start_idx:]
        time_index = range(len(plot_values))
        forecast_start = history_show

        for col, (forecasts, name, color) in enumerate(
            [
                (forecasts_normal, "NormalOutput", "blue"),
                (forecasts_quantile, "QuantileOutput", "green"),
            ]
        ):
            ax = axes[idx, col]
            ax.plot(
                time_index,
                plot_values,
                "k-",
                label="Actual",
                linewidth=1.2,
            )

            forecast = forecasts[idx]
            median = forecast.median
            lower = forecast.quantile(0.1)
            upper = forecast.quantile(0.9)

            forecast_idx = range(
                forecast_start, forecast_start + prediction_length
            )
            ax.plot(
                forecast_idx,
                median,
                color=color,
                linewidth=1.5,
                label="P50 (median)",
            )
            ax.fill_between(
                forecast_idx,
                lower,
                upper,
                alpha=0.25,
                color=color,
                label="P10-P90",
            )

            ax.axvline(
                x=forecast_start, color="gray", linestyle="--", alpha=0.7
            )
            if idx == 0:
                ax.set_title(f"DeepAR + {name}", fontsize=12)
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"Series {idx}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def main():
    seed_everything(42)

    print("=" * 60)
    print("DeepAR Comparison: NormalOutput vs QuantileOutput")
    print("=" * 60)

    freq = "H"
    prediction_length = 24
    num_epochs = 20

    print("\nGenerating synthetic sine wave dataset...")
    train_ds, test_ds = create_sine_dataset(
        num_series=20,
        length=200,
        prediction_length=prediction_length,
        freq=freq,
    )

    # Train and evaluate Normal model
    metrics_normal, forecasts_normal, tss = train_and_evaluate(
        train_ds,
        test_ds,
        NormalOutput(),
        freq=freq,
        prediction_length=prediction_length,
        num_epochs=num_epochs,
        model_name="NormalOutput",
    )

    # Train and evaluate Quantile model
    metrics_quantile, forecasts_quantile, _ = train_and_evaluate(
        train_ds,
        test_ds,
        QuantileOutput(quantiles=[0.1, 0.5, 0.9]),
        freq=freq,
        prediction_length=prediction_length,
        num_epochs=num_epochs,
        model_name="QuantileOutput",
    )

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(
        f"{'Metric':<15} {'NormalOutput':<15} {'QuantileOutput':<15} {'Better':<15}"
    )
    print("-" * 60)

    for metric in ["RMSE", "MAE", "ND", "QL_0.1", "QL_0.5", "QL_0.9"]:
        n_val = metrics_normal[metric]
        q_val = metrics_quantile[metric]
        better = "Normal" if n_val < q_val else "Quantile"
        if abs(n_val - q_val) < 0.001 * max(abs(n_val), abs(q_val), 1e-9):
            better = "Tie"
        print(f"{metric:<15} {n_val:<15.4f} {q_val:<15.4f} {better:<15}")

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_forecasts(
        forecasts_normal,
        forecasts_quantile,
        tss,
        prediction_length,
        num_plots=4,
        output_file="deepar_quantile_comparison.png",
    )

    print("\nComparison complete!")


if __name__ == "__main__":
    main()

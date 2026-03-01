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
DeepAR on Electricity: Student-t Distribution vs Quantile Regression

Compares DeepAR with StudentTOutput (NLL loss) against DeepAR with
QuantileOutput (pinball loss at P10/P50/P90) on the electricity_nips dataset.

Produces:
  - A GluonTS Evaluator metrics table (printed & saved as CSV)
  - A comparison plot of 3 randomly selected time series

Usage:
    python examples/deepar_electricity_studentt_vs_quantile.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple

from lightning import seed_everything

from gluonts.dataset.repository import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.distributions.quantile_output import QuantileOutput

# Workaround for PyTorch 2.6+ checkpoint loading with Lightning
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def train_and_evaluate(
    train_dataset,
    test_dataset,
    distr_output,
    freq: str,
    prediction_length: int,
    num_epochs: int = 10,
    model_name: str = "Model",
) -> Tuple[dict, List, List]:
    """Train a DeepAR model and return GluonTS Evaluator metrics."""
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
        num_batches_per_epoch=100,
        trainer_kwargs={
            "max_epochs": num_epochs,
            "enable_progress_bar": True,
            "enable_model_summary": False,
        },
    )

    predictor = estimator.train(train_dataset)

    forecast_it, ts_it = make_evaluation_predictions(
        test_dataset, predictor=predictor, num_samples=100
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator(quantiles=(0.1, 0.5, 0.9))
    agg_metrics, item_metrics = evaluator(
        iter(tss),
        iter(forecasts),
        num_series=len(forecasts),
    )

    return agg_metrics, forecasts, tss


def print_metrics_table(
    metrics_st: dict,
    metrics_qt: dict,
):
    """Print a side-by-side metrics comparison table."""
    rows = [
        ("RMSE", "RMSE"),
        ("NRMSE", "NRMSE"),
        ("ND", "ND"),
        ("MAPE", "MAPE"),
        ("sMAPE", "sMAPE"),
        ("mean_wQuantileLoss", "mean_wQuantileLoss"),
        ("wQuantileLoss[0.1]", "wQuantileLoss[0.1]"),
        ("wQuantileLoss[0.5]", "wQuantileLoss[0.5]"),
        ("wQuantileLoss[0.9]", "wQuantileLoss[0.9]"),
        ("Coverage[0.1]", "Coverage[0.1]"),
        ("Coverage[0.5]", "Coverage[0.5]"),
        ("Coverage[0.9]", "Coverage[0.9]"),
        ("MAE_Coverage", "MAE_Coverage"),
    ]

    header = f"{'Metric':<28} {'Student-t':<14} {'Quantile':<14} {'Better':<10}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("GluonTS Evaluation: Student-t vs Quantile Output")
    print("=" * len(header))
    print(header)
    print(sep)

    table_rows = []
    for label, key in rows:
        st_val = metrics_st.get(key, float("nan"))
        qt_val = metrics_qt.get(key, float("nan"))

        # For Coverage metrics, "better" means closer to nominal level
        if key.startswith("Coverage"):
            better = "-"
        elif key == "MAE_Coverage":
            better = (
                "Student-t" if st_val < qt_val else "Quantile"
            )
        else:
            better = "Student-t" if st_val < qt_val else "Quantile"
            if abs(st_val - qt_val) < 0.001 * max(abs(st_val), abs(qt_val), 1e-9):
                better = "Tie"

        print(f"{label:<28} {st_val:<14.4f} {qt_val:<14.4f} {better:<10}")
        table_rows.append(
            {"Metric": label, "Student-t": st_val, "Quantile": qt_val}
        )

    return pd.DataFrame(table_rows)


def plot_comparison(
    forecasts_st: List,
    forecasts_qt: List,
    tss: List,
    prediction_length: int,
    series_indices: List[int],
    output_file: str = "deepar_electricity_studentt_vs_quantile.png",
):
    """Side-by-side plot of Student-t vs Quantile forecasts for selected series."""
    num_plots = len(series_indices)
    fig, axes = plt.subplots(
        num_plots, 2, figsize=(16, 3.5 * num_plots), sharey="row"
    )
    if num_plots == 1:
        axes = axes.reshape(1, -1)

    for row, ts_idx in enumerate(series_indices):
        ts = tss[ts_idx]
        ts_values = ts.values.flatten()

        history_show = min(120, len(ts_values) - prediction_length)
        start_idx = len(ts_values) - prediction_length - history_show
        plot_values = ts_values[start_idx:]
        time_index = range(len(plot_values))
        forecast_start = history_show
        forecast_idx = range(forecast_start, forecast_start + prediction_length)

        for col, (forecasts, name, color) in enumerate(
            [
                (forecasts_st, "Student-t", "tab:blue"),
                (forecasts_qt, "Quantile", "tab:green"),
            ]
        ):
            ax = axes[row, col]
            forecast = forecasts[ts_idx]

            # Actual values
            ax.plot(
                time_index, plot_values, "k-", label="Actual", linewidth=1.0, alpha=0.8
            )

            # Median (P50)
            median = forecast.quantile(0.5)
            ax.plot(
                forecast_idx, median, color=color, linewidth=1.8, label="P50"
            )

            # P10-P90 band
            p10 = forecast.quantile(0.1)
            p90 = forecast.quantile(0.9)
            ax.fill_between(
                forecast_idx, p10, p90, alpha=0.25, color=color, label="P10-P90"
            )

            ax.axvline(x=forecast_start, color="gray", linestyle="--", alpha=0.6)

            if row == 0:
                ax.set_title(f"DeepAR + {name}", fontsize=13, fontweight="bold")
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"Series #{ts_idx}", fontsize=10)
            if row == num_plots - 1:
                ax.set_xlabel("Time Step")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def main():
    seed_everything(42)

    print("=" * 62)
    print("DeepAR on Electricity: Student-t vs Quantile Output (P10/P50/P90)")
    print("=" * 62)

    # Load dataset
    print("\nLoading electricity_nips dataset...")
    dataset = get_dataset("electricity_nips", regenerate=False)

    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    num_train_series = len(list(dataset.train))

    print(f"  Frequency: {freq}")
    print(f"  Prediction length: {prediction_length}")
    print(f"  Number of time series: {num_train_series}")

    num_epochs = 10

    # --- Student-t ---
    metrics_st, forecasts_st, tss = train_and_evaluate(
        dataset.train,
        dataset.test,
        StudentTOutput(),
        freq=freq,
        prediction_length=prediction_length,
        num_epochs=num_epochs,
        model_name="StudentTOutput",
    )

    # --- Quantile ---
    metrics_qt, forecasts_qt, _ = train_and_evaluate(
        dataset.train,
        dataset.test,
        QuantileOutput(quantiles=[0.1, 0.5, 0.9]),
        freq=freq,
        prediction_length=prediction_length,
        num_epochs=num_epochs,
        model_name="QuantileOutput(P10/P50/P90)",
    )

    # Metrics table
    df = print_metrics_table(metrics_st, metrics_qt)
    csv_path = "deepar_electricity_studentt_vs_quantile_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")

    # Pick 3 random series for plotting
    rng = np.random.RandomState(123)
    series_indices = sorted(rng.choice(len(tss), size=3, replace=False).tolist())
    print(f"\nPlotting series: {series_indices}")

    plot_comparison(
        forecasts_st,
        forecasts_qt,
        tss,
        prediction_length,
        series_indices,
        output_file="deepar_electricity_studentt_vs_quantile.png",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()

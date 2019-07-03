import pandas as pd
import json
from pathlib import Path
import os

from generate_evaluations import metrics_persisted

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


def collect_results():
    evals = []
    for evaluation_file in dir_path.glob("*/*.json"):
        with open(evaluation_file, "r") as f:
            evals.append(json.load(f))
    return pd.DataFrame(evals)


def to_markdown(df: pd.DataFrame, float_format='%.3f') -> str:
    # adapted from:
    # https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
    return os.linesep.join(
        [
            '|'.join(df.columns),
            '|'.join(4 * '-' for _ in df.columns),
            df.to_csv(
                sep='|', index=False, header=False, float_format=float_format
            ),
        ]
    ).replace('|', ' | ')


results_df = collect_results()

# copy-paste the results in `evaluations/README.md`
for metric in metrics_persisted:
    print(f"## {metric}\n")

    pivot_df = results_df.pivot_table(
        index="estimator", columns="dataset", values=metric
    )
    print(to_markdown(pivot_df.reset_index(level=0)))

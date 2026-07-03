import importlib.util
from pathlib import Path

import numpy as np

from gluonts.dataset.common import ListDataset


def load_grouper():
    path = (
        Path(__file__).parents[3]
        / "src"
        / "gluonts"
        / "nursery"
        / "robust-mts-attack"
        / "multivariate"
        / "datasets"
        / "grouper.py"
    )
    spec = importlib.util.spec_from_file_location(
        "robust_mts_attack_grouper", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Grouper


def test_grouper_splits_rolling_test_data_before_stacking():
    dataset = ListDataset(
        [
            {"start": "2014-09-07", "target": [1, 2, 3, 4]},
            {"start": "2014-09-07", "target": [5, 6, 7, 8]},
            {"start": "2014-09-08", "target": [0, 1, 2, 3]},
            {"start": "2014-09-08", "target": [4, 5, 6, 7]},
        ],
        freq="1D",
    )

    Grouper = load_grouper()
    grouped_data = list(Grouper(num_test_dates=2)(dataset))

    np.testing.assert_array_equal(
        grouped_data[0]["target"], np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    )
    np.testing.assert_array_equal(
        grouped_data[1]["target"],
        np.array([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
    )

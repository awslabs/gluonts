import logging
import os
from functools import partial
from pathlib import Path

from gluonts.dataset.common import load_datasets, TrainDatasets
from gluonts.dataset.repository.m4 import generate_m4_dataset

m4_freq = "Hourly"
pandas_freq = "H"
dataset_path = Path(f'm4-{m4_freq}')
prediction_length = 48


dataset_recipes = {
    # each recipe generates a dataset given a path
    "m4_hourly": partial(
        generate_m4_dataset,
        m4_freq="Hourly",
        pandas_freq="H",
        prediction_length=48,
    ),
    "m4_daily": partial(
        generate_m4_dataset,
        m4_freq="Daily",
        pandas_freq="D",
        prediction_length=14,
    ),
    "m4_weekly": partial(
        generate_m4_dataset,
        m4_freq="Weekly",
        pandas_freq="W",
        prediction_length=13,
    ),
    "m4_monthly": partial(
        generate_m4_dataset,
        m4_freq="Monthly",
        pandas_freq="M",
        prediction_length=18,
    ),
    "m4_quarterly": partial(
        generate_m4_dataset,
        m4_freq="Quarterly",
        pandas_freq="3M",
        prediction_length=8,
    ),
    "m4_yearly": partial(
        generate_m4_dataset,
        m4_freq="Yearly",
        pandas_freq="12M",
        prediction_length=6,
    ),
}


def get_dataset(dataset_name: str, regenerate: bool = False) -> TrainDatasets:
    """
    Parameters
    ----------
    dataset_name
        name of the dataset, for instance "m4_hourly"
    regenerate
        whether to regenerate the dataset even if a local file is present. If this flag is False and the
        file is present, the dataset will not be downloaded again.

    Returns
    -------
        dataset obtained by either downloading or reloading from local file.
    """
    assert (
        dataset_name in dataset_recipes.keys()
    ), f"{dataset_name} is not present, please choose one from {dataset_recipes.keys()}."
    dataset_path = Path(dataset_name)

    dataset_recipe = dataset_recipes[dataset_name]

    if not os.path.exists(dataset_path) or regenerate:
        logging.info(f"downloading and processing {dataset_name}")
        dataset_recipe(dataset_path=dataset_path)
    else:
        logging.info(
            f"using dataset already processed in path {dataset_path}."
        )

    return load_datasets(
        metadata=dataset_path,
        train=dataset_path / 'train',
        test=dataset_path / 'test',
    )


if __name__ == '__main__':
    for dataset in dataset_recipes.keys():
        print(f"generate {dataset}")
        get_dataset(dataset, regenerate=True)

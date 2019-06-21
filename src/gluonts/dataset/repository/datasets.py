import logging
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

from gluonts.dataset.common import TrainDatasets, load_datasets
from gluonts.dataset.repository._lstnet import generate_lstnet_dataset
from gluonts.dataset.repository._m4 import generate_m4_dataset
from gluonts.support.util import get_download_path

m4_freq = "Hourly"
pandas_freq = "H"
dataset_path = Path(f'm4-{m4_freq}')
prediction_length = 48


dataset_recipes = OrderedDict(
    {
        # each recipe generates a dataset given a path
        "exchange_rate": partial(
            generate_lstnet_dataset, dataset_name="exchange_rate"
        ),
        "solar-energy": partial(
            generate_lstnet_dataset, dataset_name="solar-energy"
        ),
        "electricity": partial(
            generate_lstnet_dataset, dataset_name="electricity"
        ),
        "traffic": partial(generate_lstnet_dataset, dataset_name="traffic"),
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
)


dataset_names = list(dataset_recipes.keys())

default_dataset_path = get_download_path() / "datasets"


def get_dataset(
    dataset_name: str,
    regenerate: bool = False,
    path: Path = default_dataset_path,
) -> TrainDatasets:
    """
    Parameters
    ----------
    dataset_name
        name of the dataset, for instance "m4_hourly"
    regenerate
        whether to regenerate the dataset even if a local file is present. If this flag is False and the
        file is present, the dataset will not be downloaded again.
    path
        where the dataset should be saved
    Returns
    -------
        dataset obtained by either downloading or reloading from local file.
    """
    assert (
        dataset_name in dataset_recipes.keys()
    ), f"{dataset_name} is not present, please choose one from {dataset_recipes.keys()}."

    dataset_path = path / dataset_name

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

    for dataset in dataset_names:
        print(f"generate {dataset}")
        ds = get_dataset(dataset, regenerate=True)
        print(ds.metadata)
        print(sum(1 for _ in list(iter(ds.train))))

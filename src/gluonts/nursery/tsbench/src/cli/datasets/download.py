from functools import partial
from pathlib import Path
from typing import Optional
import click
from tsbench.analysis.utils import num_fitting_processes, run_parallel
from tsbench.config import DATASET_REGISTRY
from tsbench.constants import DEFAULT_DATA_PATH
from ._main import datasets


@datasets.command(short_help="Download and preprocess datasets.")
@click.option(
    "--dataset",
    type=str,
    default=None,
    help="The dataset to compute basic statistics for. "
    "If not provided, computes statistics for all datasets.",
)
@click.option(
    "--path",
    type=str,
    default=DEFAULT_DATA_PATH,
    show_default=True,
    help="The path where the datasets should be downloaded to.",
)
def download(dataset: Optional[str], path: str):
    """
    Downloads and preprocesses either a single dataset or all datasets in the registry.
    """
    base = Path(path)

    if dataset is not None:
        dataset_cls = DATASET_REGISTRY[dataset](base)
        dataset_cls.generate()
        dataset_cls.prepare()
        return

    # Start off by downloading an M3 dataset
    dataset_cls = DATASET_REGISTRY["m3_monthly"](base)
    dataset_cls.generate()
    dataset_cls.prepare()

    # Then, we can download the rest in parallel (by preloading, we don't download the M3 data in
    # parallel)
    run_parallel(
        partial(_download_dataset, base=base),
        list(DATASET_REGISTRY.keys()),
        num_processes=min(
            num_fitting_processes(cpus_per_process=1, memory_per_process=8),
            len(DATASET_REGISTRY),
        ),
    )


def _download_dataset(name: str, base: Path):
    dataset_cls = DATASET_REGISTRY[name](base)
    dataset_cls.generate()
    dataset_cls.prepare()

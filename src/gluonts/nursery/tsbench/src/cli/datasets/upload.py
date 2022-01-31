import click
from tqdm.auto import tqdm
from tsbench.config import DATASET_REGISTRY
from tsbench.constants import DEFAULT_DATA_PATH
from tsbench.evaluations.aws import default_session
from tsbench.evaluations.aws.s3 import upload_directory
from ._main import datasets


@datasets.command(short_help="Upload locally available datasets to S3.")
@click.option(
    "--bucket",
    required=True,
    help="The S3 bucket to upload the dataset to.",
)
@click.option(
    "--path",
    type=str,
    default=DEFAULT_DATA_PATH,
    show_default=True,
    help="The local path where the datasets are stored.",
)
@click.option(
    "--prefix",
    type=str,
    default="data",
    show_default=True,
    help="The prefix in the S3 bucket.",
)
def upload(path: str, bucket: str, prefix: str):
    """
    Uploads the data for all datasets available locally to an S3 bucket.
    """
    s3 = default_session().client("s3")
    for config in tqdm(DATASET_REGISTRY.values()):
        upload_directory(s3, path / config.name(), bucket, f"{prefix}/{config.name()}")

from pathlib import Path
import click
from tsbench.constants import DEFAULT_EVALUATIONS_PATH
from tsbench.utils import compress_directory
from ._main import evaluations


@evaluations.command(short_help="Archive metrics of all evaluations into a single file.")
@click.option(
    "--evaluations_path",
    type=click.Path(exists=True),
    default=DEFAULT_EVALUATIONS_PATH,
    help="The directory where TSBench evaluations are stored.",
)
@click.option(
    "--archive_path",
    type=click.Path(),
    default=Path.home() / "archive",
    help="The directory to which to write the compressed files.",
)
def archive(evaluations_path: str, archive_path: str):
    """
    Archives the metrics of all evaluations found in the provided directory into a single file.
    This is most probably only necessary for publishing the metrics in an easier format.
    """
    source = Path(evaluations_path)
    target = Path(archive_path)
    target.mkdir(parents=True, exist_ok=True)

    # First, tar all the metadata
    print("Compressing metrics...")
    compress_directory(
        source,
        target / "metrics.tar.gz",
        include={"config.json", "performance.json"},
    )

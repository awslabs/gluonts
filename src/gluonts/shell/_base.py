# Standard library imports
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Iterator

# First-party imports
from gluonts.dataset.repository import DatasetRepository


class PathsEnvironment(NamedTuple):
    config: Path = Path('/opt/ml/input/config')
    data: Path = Path('/opt/ml/input/data')
    model: Path = Path('/opt/ml/model')
    output: Path = Path('/opt/ml/output')

    @property
    def output_data(self) -> Path:
        return self.output / 'data'

    def makedirs(self) -> None:
        self.config.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.model.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.output_data.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_environment(
    hyperparameters: dict, dataset_name: str, repository: DatasetRepository
) -> Iterator[PathsEnvironment]:
    info, _, _ = repository.get_from_name(dataset_name)
    hyperparameters = {
        **hyperparameters,
        'freq': info.metadata.time_granularity,
    }

    with tempfile.TemporaryDirectory(prefix='gluonts-temp') as base:
        paths = PathsEnvironment(
            config=Path(base) / 'config',
            data=Path(f'{repository.dataset_local_path}/{dataset_name}'),
            model=Path(base) / 'model',
            output=Path(base) / 'output',
        )
        paths.makedirs()

        with open(str(paths.config / 'hyperparameters.json'), mode='w') as fp:
            json.dump(hyperparameters, fp, indent=2, sort_keys=True)

        yield paths

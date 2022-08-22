from dataclasses import dataclass

import pytest

from gluonts.exceptions import GluonTSDataError
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

@dataclass
class CustomDataset:
    data: list

    def __iter__(self):
        for el in self.data:
            yield el


@pytest.mark.parametrize(
    "dataset", [CustomDataset([])]
)
def test_deepar_no_batches(dataset):
    estimator = DeepAREstimator(
        prediction_length=10,
        freq="H",
        trainer=Trainer(epochs=1, num_batches_per_epoch=1),
    )

    with pytest.raises(GluonTSDataError):
        estimator.train(dataset)

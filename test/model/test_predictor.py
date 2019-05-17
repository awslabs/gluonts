# Third-party imports
import numpy as np

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.predictor import ParallelizedPredictor, Localizer
from gluonts.model.testutil import IdentityPredictor, MeanEstimator


def test_parallelized_predictor():
    dataset = ListDataset(
        data_iter=[
            {'start': '2012-01-01', 'target': (np.zeros(20) + i).tolist()}
            for i in range(300)
        ],
        freq='1H',
    )

    base_predictor = IdentityPredictor(
        freq='1H', prediction_length=10, num_samples=100
    )

    predictor = ParallelizedPredictor(
        base_predictor=base_predictor, num_workers=10, chunk_size=2
    )

    predictions = list(base_predictor.predict(dataset))
    parallel_predictions = list(predictor.predict(dataset))

    assert len(predictions) == len(parallel_predictions)

    for p, pp in zip(predictions, parallel_predictions):
        assert np.all(p.samples == pp.samples)
        assert np.all(p.index == pp.index)


def test_localizer():
    dataset = ListDataset(
        data_iter=[
            {
                'start': '2012-01-01',
                'target': (np.zeros(20) + i * 0.1 + 0.01),
                'id': f'{i}',
            }
            for i in range(3)
        ],
        freq='1H',
    )

    estimator = MeanEstimator(prediction_length=10, freq='1H', num_samples=50)

    local_pred = Localizer(estimator=estimator)
    agg_metrics, _ = backtest_metrics(
        train_dataset=None, test_dataset=dataset, forecaster=local_pred
    )

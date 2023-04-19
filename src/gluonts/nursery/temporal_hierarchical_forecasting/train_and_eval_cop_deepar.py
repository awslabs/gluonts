from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx.trainer import Trainer

from gluonts.nursery.temporal_hierarchical_forecasting.model.cop_deepar import (
    COPDeepAREstimator,
)
from gluonts.nursery.temporal_hierarchical_forecasting.eval.evaluation import (
    evaluate_predictor,
)


dataset = get_dataset("exchange_rate", regenerate=False)

estimator = COPDeepAREstimator(
    freq=dataset.metadata.freq,
    prediction_length=dataset.metadata.prediction_length,
    base_estimator_name="DeepAREstimatorForCOP",
    base_estimator_hps={},
    trainer=Trainer(
        epochs=1,
        hybridize=False,
    ),
)

predictor = estimator.train(dataset.train)

results = evaluate_predictor(
    predictor=predictor,
    test_dataset=dataset.test,
    evaluate_all_levels=EVALUATE_ALL_LEVELS,
    freq=dataset.metadata.freq,
)

print(results)

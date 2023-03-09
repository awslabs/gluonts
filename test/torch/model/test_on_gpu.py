import pandas as pd
import numpy as np
import torch
import pytest

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import Evaluator


FREQ = "D"
PRED_LENGTH = 5
TRAINER_KARGS = {"max_epochs": 1, "accelerator": "gpu", "devices": 1}


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="only run the test on GPU"
)
@pytest.mark.parametrize(
    "model",
    [
        SimpleFeedForwardEstimator(
            prediction_length=PRED_LENGTH, trainer_kwargs=TRAINER_KARGS
        ),
        DeepAREstimator(
            freq=FREQ,
            prediction_length=PRED_LENGTH,
            trainer_kwargs=TRAINER_KARGS,
        ),
    ],
)
def test_torch_model_on_gpu(
    model: PyTorchLightningEstimator,
):
    n = 100
    date = "2015-04-07 00:00:00"

    df = pd.DataFrame(
        np.random.randn(n).astype(np.float32),
        index=pd.period_range(date, periods=n, freq=FREQ),
    )
    dataset = PandasDataset(df, target=0)

    predictor = model.train(dataset)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset, predictor=predictor, num_samples=100
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(
        ts_it,
        forecast_it,
    )

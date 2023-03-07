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


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="only run the test on GPU"
)
@pytest.mark.parametrize(
    "torch_model_cls", [SimpleFeedForwardEstimator, DeepAREstimator]
)
def test_torch_model_on_gpu(
    torch_model_cls: PyTorchLightningEstimator,
):

    n = 100
    freq = "D"
    date = "2015-04-07 00:00:00"
    pred_length = 5

    df = pd.DataFrame(
        np.random.randn(n).astype(np.float32),
        index=pd.period_range(date, periods=n, freq=freq),
    )
    dataset = PandasDataset(df, target=0)

    train_args = {"max_epochs": 1, "accelerator": "gpu", "devices": 1}

    if torch_model_cls == SimpleFeedForwardEstimator:
        model = torch_model_cls(
            prediction_length=pred_length, trainer_kwargs=train_args
        )
    else:
        model = torch_model_cls(
            freq=freq, prediction_length=pred_length, trainer_kwargs=train_args
        )

    predictor = model.train(dataset)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset, predictor=predictor, num_samples=100
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(
        ts_it,
        forecast_it,
    )

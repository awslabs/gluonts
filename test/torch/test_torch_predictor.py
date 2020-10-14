import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.predictor import Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import InstanceSplitter, ExpectedNumInstanceSampler


class RandomNetwork(nn.Module):
    @validated()
    def __init__(self, prediction_length: int, context_length: int,) -> None:
        super().__init__()
        assert prediction_length > 0
        assert context_length > 0
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.net = nn.Linear(context_length, prediction_length)
        torch.nn.init.uniform_(self.net.weight, -1.0, 1.0)

    def forward(self, context):
        assert context.shape[-1] == self.context_length
        out = self.net(context)
        return out.unsqueeze(1)


def test_pytorch_predictor_serde():
    context_length = 20
    prediction_length = 5

    transformation = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        train_sampler=ExpectedNumInstanceSampler(num_instances=1),
        past_length=context_length,
        future_length=prediction_length,
    )

    pred_net = RandomNetwork(
        prediction_length=prediction_length, context_length=context_length
    )

    predictor = PyTorchPredictor(
        prediction_length=prediction_length,
        freq="1H",
        input_names=["past_target"],
        prediction_net=pred_net,
        batch_size=16,
        input_transform=transformation,
        device=None,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        predictor.serialize(Path(temp_dir))
        predictor_exp = Predictor.deserialize(Path(temp_dir))

    def test_data_generator():
        for _ in range(20):
            yield {
                FieldName.START: pd.Timestamp(
                    "2020-01-01 00:00:00", freq="1H"
                ),
                FieldName.TARGET: np.random.uniform(size=(100,)).astype("f"),
            }

    test_data = list(test_data_generator())

    for f, f_exp in zip(
        predictor.predict(test_data), predictor_exp.predict(test_data)
    ):
        assert np.allclose(f.samples, f_exp.samples)

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Tests for the training-failure recovery path in
``PyTorchLightningEstimator.train_model``.

The recovery ``try/except`` around ``trainer.fit`` exists to survive the
numerical-instability failures training can hit late in a run (e.g. the
``StudentTOutput`` NaN ``ValueError`` from #3265). It must recover from those
by loading the best checkpoint, but it must NOT swallow unrelated failures
(CUDA OOM, misconfiguration, ``KeyboardInterrupt``) -- doing so would return a
stale checkpoint from a much earlier epoch and turn a hard failure into a
silent accuracy regression (issue #3295).
"""

from typing import Iterable, Optional
from unittest import mock

import pytest
import torch.nn as nn

from gluonts.torch.model.estimator import PyTorchLightningEstimator


class _DummyEstimator(PyTorchLightningEstimator):
    """Minimal concrete estimator whose data plumbing is stubbed out so the
    only thing exercised is the recovery ``try/except`` around ``fit``."""

    def __init__(self):
        super().__init__(trainer_kwargs={})

    def create_transformation(self):
        from gluonts.transform import Identity

        return Identity()

    def create_lightning_module(self) -> nn.Module:
        module = nn.Linear(1, 1)
        # train_model calls create_predictor(transformation, module); give the
        # module the attribute access that path needs without real training.
        return module

    def create_training_data_loader(self, data, module, **kwargs) -> Iterable:
        return []

    def create_validation_data_loader(
        self, data, module, **kwargs
    ) -> Iterable:
        return []

    def create_predictor(self, transformation, module):
        return mock.MagicMock(name="predictor")


def _run_with_fit_raising(exc: BaseException, best_model_path: str):
    """Drive train_model with a patched pl.Trainer whose fit() raises ``exc``
    and a ModelCheckpoint reporting ``best_model_path``."""
    est = _DummyEstimator()

    fake_checkpoint = mock.MagicMock()
    fake_checkpoint.best_model_path = best_model_path

    fake_trainer = mock.MagicMock()
    fake_trainer.fit.side_effect = exc

    with (
        mock.patch(
            "gluonts.torch.model.estimator.pl.callbacks.ModelCheckpoint",
            return_value=fake_checkpoint,
        ),
        mock.patch(
            "gluonts.torch.model.estimator.pl.Trainer",
            return_value=fake_trainer,
        ),
        mock.patch(
            "gluonts.torch.model.estimator.torch.load",
            return_value={"state_dict": nn.Linear(1, 1).state_dict()},
        ),
    ):
        return est.train_model(training_data=[{"target": [1.0, 2.0, 3.0]}])


def test_recovers_from_valueerror_when_checkpoint_exists():
    # A ValueError (the targeted numerical-instability case) with a checkpoint
    # available must be recovered from, not re-raised.
    out = _run_with_fit_raising(
        ValueError("nan loss"), best_model_path="/tmp/best.ckpt"
    )
    assert out is not None


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("cuda oom proxy"),
        KeyboardInterrupt(),
    ],
)
def test_unrelated_failures_propagate_even_with_checkpoint(exc):
    # An unrelated failure must propagate even when a checkpoint exists,
    # instead of being masked and returning a stale early-epoch checkpoint.
    with pytest.raises(type(exc)):
        _run_with_fit_raising(exc, best_model_path="/tmp/best.ckpt")


def test_reraises_when_no_checkpoint_available():
    with pytest.raises(ValueError):
        _run_with_fit_raising(ValueError("nan loss"), best_model_path="")

# Standard library imports
import tempfile
from pathlib import Path

# Third-party imports
import pytest

# First-party imports
from gluonts import time_feature
from gluonts.core.serde import load_code
from gluonts.dataset.artificial import constant_dataset
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.npts import NPTSEstimator
from gluonts.model.predictor import Predictor
from gluonts.model.seasonal_naive import SeasonalNaiveEstimator
from gluonts.model.seq2seq import (
    MQCNNEstimator,
    MQRNNEstimator,
    Seq2SeqEstimator,
)
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

dataset_info, train_ds, test_ds = constant_dataset()
freq = dataset_info.metadata.freq
prediction_length = dataset_info.prediction_length
cardinality = int(dataset_info.metadata.feat_static_cat[0].cardinality)
# FIXME: Should time features should not be needed for GP
time_features = [time_feature.DayOfWeek(), time_feature.HourOfDay()]
num_eval_samples = 2
epochs = 1


def seq2seq_base(seq2seq_model, hybridize: bool = True, batches_per_epoch=1):
    return (
        seq2seq_model,
        dict(
            ctx='cpu',
            epochs=epochs,
            learning_rate=1e-2,
            hybridize=hybridize,
            prediction_length=prediction_length,
            context_length=prediction_length,
            num_eval_samples=num_eval_samples,
            num_batches_per_epoch=batches_per_epoch,
            quantiles=[0.1, 0.5, 0.9],
            use_symbol_block_predictor=True,
        ),
    )


def mqcnn_estimator(hybridize: bool = True, batches_per_epoch=1):
    return seq2seq_base(MQCNNEstimator, hybridize, batches_per_epoch)


def mqrnn_estimator(hybridize: bool = True, batches_per_epoch=1):
    return seq2seq_base(MQRNNEstimator, hybridize, batches_per_epoch)


def npts_estimator():
    return (
        NPTSEstimator,
        dict(
            kernel_type='uniform',
            use_default_features=True,
            prediction_length=prediction_length,
            num_eval_samples=num_eval_samples,
        ),
    )


def simple_seq2seq_estimator(hybridize: bool = True, batches_per_epoch=1):
    return seq2seq_base(Seq2SeqEstimator, hybridize, batches_per_epoch)


def simple_feedforward_estimator(hybridize: bool = True, batches_per_epoch=1):
    return (
        SimpleFeedForwardEstimator,
        dict(
            ctx='cpu',
            epochs=epochs,
            learning_rate=1e-2,
            hybridize=hybridize,
            num_hidden_dimensions=[3],
            prediction_length=prediction_length,
            num_eval_samples=num_eval_samples,
            num_batches_per_epoch=batches_per_epoch,
            use_symbol_block_predictor=True,
        ),
    )


def gp_estimator(hybridize: bool = True, batches_per_epoch=1):
    return (
        GaussianProcessEstimator,
        dict(
            ctx='cpu',
            epochs=epochs,
            learning_rate=1e-2,
            hybridize=hybridize,
            prediction_length=prediction_length,
            cardinality=cardinality,
            num_eval_samples=num_eval_samples,
            num_batches_per_epoch=batches_per_epoch,
            time_features=time_features,
            use_symbol_block_predictor=False,
            # FIXME: test_shell fails with use_symbol_block_predictor=True
            # FIXME and float_type = np.float64
        ),
    )


def deepar_estimator(hybridize: bool = True, batches_per_epoch=1):
    return (
        DeepAREstimator,
        dict(
            ctx='cpu',
            epochs=epochs,
            learning_rate=1e-2,
            hybridize=hybridize,
            num_cells=2,
            num_layers=1,
            prediction_length=prediction_length,
            context_length=2,
            num_eval_samples=2,
            num_batches_per_epoch=batches_per_epoch,
            use_symbol_block_predictor=False,
        ),
    )


def seasonal_estimator():
    return SeasonalNaiveEstimator, dict(prediction_length=prediction_length)


@pytest.mark.timeout(10)  # DeepAR occasionally fails with the 5 second timeout
@pytest.mark.parametrize(
    "Estimator, hyperparameters, accuracy",
    [
        simple_feedforward_estimator(batches_per_epoch=200) + (0.3,),
        gp_estimator(batches_per_epoch=200) + (0.2,),
        npts_estimator() + (0.0,),
        # deepar_estimator(batches_per_epoch=50) + (1.5,), # large value as this test is breaking frequently
        seasonal_estimator() + (0.0,),
        # mqcnn_estimator(batches_per_epoch=200) + (0.2,),
        # mqrnn_estimator(batches_per_epoch=200) + (0.2,),
    ],
)
def test_accuracy(Estimator, hyperparameters, accuracy):
    estimator = Estimator.from_hyperparameters(freq=freq, **hyperparameters)
    agg_metrics, item_metrics = backtest_metrics(
        train_dataset=train_ds, test_dataset=test_ds, forecaster=estimator
    )

    assert agg_metrics['ND'] <= accuracy


@pytest.mark.parametrize(
    "Estimator, hyperparameters",
    [
        simple_feedforward_estimator(),
        deepar_estimator(),
        npts_estimator(),
        seasonal_estimator(),
        mqcnn_estimator(),
        mqrnn_estimator(),
        gp_estimator(),
    ],
)
def test_repr(Estimator, hyperparameters):
    estimator = Estimator.from_hyperparameters(freq=freq, **hyperparameters)
    assert repr(estimator) == repr(load_code(repr(estimator)))


@pytest.mark.parametrize(
    "Estimator, hyperparameters",
    [
        simple_feedforward_estimator(hybridize=True),
        deepar_estimator(hybridize=True),
        mqcnn_estimator(hybridize=True),
        mqrnn_estimator(hybridize=True),
        gp_estimator(hybridize=True),
    ],
)
def test_hybridize(Estimator, hyperparameters):
    estimator = Estimator.from_hyperparameters(freq=freq, **hyperparameters)
    backtest_metrics(
        train_dataset=train_ds, test_dataset=test_ds, forecaster=estimator
    )


@pytest.mark.parametrize(
    "Estimator, hyperparameters",
    [
        simple_feedforward_estimator(),
        deepar_estimator(),
        npts_estimator(),
        seasonal_estimator(),
        mqcnn_estimator(),
        mqrnn_estimator(),
        gp_estimator(),
    ],
)
def test_serialize(Estimator, hyperparameters):
    estimator = Estimator.from_hyperparameters(freq=freq, **hyperparameters)
    with tempfile.TemporaryDirectory() as temp_dir:
        predictor_act = estimator.train(train_ds)
        predictor_act.serialize(Path(temp_dir))
        predictor_exp = Predictor.deserialize(Path(temp_dir))
        assert predictor_act == predictor_exp

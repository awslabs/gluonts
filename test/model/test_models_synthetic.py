# Third-party imports
import pytest

# First-party imports
from gluonts.dataset.artificial import default_synthetic
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.wavenet import WaveNetEstimator

dataset_info, train_ds, test_ds = default_synthetic()
freq = dataset_info.metadata.freq
prediction_length = dataset_info.prediction_length
cardinality = int(dataset_info.metadata.feat_static_cat[0].cardinality)
batch_size = 32
context_length = 2
epochs = 1


def simple_feedforward_estimator(hybridize: bool = True, batches_per_epoch=1):
    return (
        SimpleFeedForwardEstimator,
        dict(
            ctx='cpu',
            epochs=epochs,
            learning_rate=1e-2,
            batch_size=batch_size,
            hybridize=hybridize,
            num_hidden_dimensions=[3],
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            num_eval_samples=5,
            num_batches_per_epoch=batches_per_epoch,
        ),
    )


def deepar_estimator(hybridize: bool = False, batches_per_epoch=1):
    return (
        DeepAREstimator,
        dict(
            ctx='cpu',
            epochs=epochs,
            learning_rate=1e-2,
            batch_size=batch_size,
            hybridize=hybridize,
            num_cells=2,
            num_layers=1,
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            num_eval_samples=2,
            num_batches_per_epoch=batches_per_epoch,
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
            context_length=context_length,
            freq=freq,
            cardinality=cardinality,
            num_eval_samples=5,
            num_batches_per_epoch=batches_per_epoch,
        ),
    )


def wavenet_estimator(hybridize: bool = False, batches_per_epoch=1):
    return (
        WaveNetEstimator,
        dict(
            ctx='cpu',
            epochs=epochs,
            learning_rate=1e-2,
            hybridize=hybridize,
            prediction_length=prediction_length,
            freq=freq,
            cardinality=[cardinality],
            num_eval_samples=5,
            num_batches_per_epoch=batches_per_epoch,
        ),
    )


@pytest.mark.timeout(5)  # DeepAR occasionally fails the 5 second timeout
@pytest.mark.parametrize(
    "Estimator, hyperparameters, accuracy",
    [
        simple_feedforward_estimator(batches_per_epoch=1) + (10.0,),
        deepar_estimator(batches_per_epoch=1) + (10.0,),
        gp_estimator(batches_per_epoch=1) + (10.0,),
        wavenet_estimator(batches_per_epoch=10) + (10.0,),
    ],
)
def test_accuracy(Estimator, hyperparameters, accuracy):
    estimator = Estimator.from_hyperparameters(**hyperparameters)
    agg_metrics, item_metrics = backtest_metrics(
        train_dataset=train_ds, test_dataset=test_ds, forecaster=estimator
    )

    assert agg_metrics['ND'] <= accuracy

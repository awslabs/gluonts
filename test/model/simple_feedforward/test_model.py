import pytest

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator


@pytest.fixture()
def hyperparameters():
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        num_hidden_dimensions=[3],
        num_batches_per_epoch=1,
        use_symbol_block_predictor=True,
    )


@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(accuracy_test, hyperparameters, hybridize):
    hyperparameters.update(num_batches_per_epoch=200, hybridize=hybridize)

    accuracy_test(SimpleFeedForwardEstimator, hyperparameters, accuracy=0.3)


def test_repr(repr_test, hyperparameters):
    repr_test(SimpleFeedForwardEstimator, hyperparameters)


def test_serialize(serialize_test, hyperparameters):
    serialize_test(SimpleFeedForwardEstimator, hyperparameters)

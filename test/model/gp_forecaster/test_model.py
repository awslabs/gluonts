import pytest

from gluonts.model.gp_forecaster import GaussianProcessEstimator


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        cardinality=dsinfo.cardinality,
        num_batches_per_epoch=1,
        time_features=dsinfo.time_features,
        use_symbol_block_predictor=False,
        # FIXME: test_shell fails with use_symbol_block_predictor=True
        # FIXME and float_type = np.float64
    )


@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(accuracy_test, hyperparameters, hybridize):
    hyperparameters.update(num_batches_per_epoch=200, hybridize=hybridize)

    accuracy_test(GaussianProcessEstimator, hyperparameters, accuracy=0.2)


def test_repr(repr_test, hyperparameters):
    repr_test(GaussianProcessEstimator, hyperparameters)


def test_serialize(serialize_test, hyperparameters):
    serialize_test(GaussianProcessEstimator, hyperparameters)

import pytest

from gluonts.model.deep_factor import DeepFactorEstimator


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        cardinality=[dsinfo.cardinality],
        num_batches_per_epoch=1,
        use_symbol_block_predictor=False,
    )


@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(accuracy_test, hyperparameters, hybridize):
    hyperparameters.update(num_batches_per_epoch=200, hybridize=hybridize)

    accuracy_test(DeepFactorEstimator, hyperparameters, accuracy=0.3)


def test_repr(repr_test, hyperparameters):
    repr_test(DeepFactorEstimator, hyperparameters)


# TODO: Enable this test: Error:  assert <gluonts.model.predictor.RepresentableBlockPredictor object at
# TODO: 0x124701240> == <gluonts.model.predictor.RepresentableBlockPredictor object at 0x124632940>
@pytest.mark.xfail
def test_serialize(serialize_test, hyperparameters):
    serialize_test(DeepFactorEstimator, hyperparameters)

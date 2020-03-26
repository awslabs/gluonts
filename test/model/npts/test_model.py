from gluonts.model.npts import NPTSEstimator

hyperparameters = dict(kernel_type="uniform", use_default_features=True,)


def test_accuracy(accuracy_test):
    accuracy_test(NPTSEstimator, hyperparameters, accuracy=0.0)


def test_repr(repr_test):
    repr_test(NPTSEstimator, hyperparameters)


def test_serialize(serialize_test):
    serialize_test(NPTSEstimator, hyperparameters)

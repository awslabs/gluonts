# Standard library imports
import itertools

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.model.deepar._network import DeepARTrainingNetwork


def test_lagged_subsequences():
    N = 8
    T = 96
    C = 2
    lags = [1, 2, 3, 24, 48]
    I = len(lags)
    sequence = mx.nd.random.normal(shape=(N, T, C))
    S = 48

    # (batch_size, sub_seq_len, target_dim, num_lags)
    lagged_subsequences = DeepARTrainingNetwork.get_lagged_subsequences(
        F=mx.nd,
        sequence=sequence,
        sequence_length=sequence.shape[1],
        indices=lags,
        subsequences_length=S,
    )

    assert (N, S, C, I) == lagged_subsequences.shape

    # checks that lags value behave as described as in the get_lagged_subsequences contract
    for i, j, k in itertools.product(range(N), range(S), range(I)):
        assert (
            (
                lagged_subsequences[i, j, :, k]
                == sequence[i, -lags[k] - S + j, :]
            )
            .asnumpy()
            .all()
        )

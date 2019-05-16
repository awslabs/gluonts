# Third-party imports
from mxnet import nd

# First-party imports
from gluonts.block.quantile_output import QuantileLoss


def test_compute_quantile_loss() -> None:
    y_true = nd.ones(shape=(10, 10, 10))
    y_pred = nd.zeros(shape=(10, 10, 10, 2))

    quantiles = [0.5, 0.9]

    loss = QuantileLoss(quantiles)

    correct_qt_loss = [1.0, 1.8]

    for idx, q in enumerate(quantiles):
        assert (
            nd.mean(
                loss.compute_quantile_loss(
                    nd.ndarray, y_true, y_pred[:, :, :, idx], q
                )
            )
            - correct_qt_loss[idx]
            < 1e-5
        ), f"computing quantile loss at quantile {q} fails!"

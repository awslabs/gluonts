from itertools import islice

from gluonts.distribution import StudentTOutput, StudentT
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.model.deepar import DeepAREstimator
import mxnet as mx
from gluonts.trainer import Trainer


ds_info, train_ds, test_ds = constant_dataset()
freq = ds_info.metadata.freq
prediction_length = ds_info.prediction_length


def test_distribution():
    """
    Makes sure additional tensors can be accessed and have expected shapes
    """
    prediction_length = ds_info.prediction_length
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        trainer=Trainer(epochs=1, num_batches_per_epoch=1),
        distr_output=StudentTOutput(),
    )

    train_output = estimator.train_model(train_ds)

    # todo adapt loader to anomaly detection use-case
    batch_size = 2
    num_samples = 3

    training_data_loader = TrainDataLoader(
        dataset=train_ds,
        transform=train_output.transformation,
        batch_size=batch_size,
        num_batches_per_epoch=estimator.trainer.num_batches_per_epoch,
        ctx=mx.cpu(),
    )

    seq_len = 2 * ds_info.prediction_length

    for data_entry in islice(training_data_loader, 1):
        input_names = get_hybrid_forward_input_names(train_output.trained_net)

        distr = train_output.trained_net.distribution(
            *[data_entry[k] for k in input_names]
        )

        assert distr.sample(num_samples).shape == (
            num_samples,
            batch_size,
            seq_len,
        )

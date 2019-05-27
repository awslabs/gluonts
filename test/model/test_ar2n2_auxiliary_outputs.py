from itertools import islice

from gluonts.distribution import StudentTOutput, StudentT
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.model.deepar import DeepAREstimator
import mxnet as mx
from gluonts.trainer import Trainer


ds_info, train_ds, test_ds = constant_dataset()
freq = ds_info.metadata.time_granularity
prediction_length = ds_info.prediction_length


def test_shape():
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

    training_transformation, trained_net = estimator.train_model(train_ds)

    # todo adapt loader to anomaly detection use-case
    batch_size = 2
    training_data_loader = TrainDataLoader(
        dataset=train_ds,
        transform=training_transformation,
        batch_size=batch_size,
        num_batches_per_epoch=estimator.trainer.num_batches_per_epoch,
        ctx=mx.cpu(),
    )

    seq_len = 2 * ds_info.prediction_length

    for data_entry in islice(training_data_loader, 1):
        input_names = get_hybrid_forward_input_names(trained_net)

        loss, likelihoods, *distr_args = trained_net(
            *[data_entry[k] for k in input_names]
        )

        distr = StudentT(*distr_args)

        assert likelihoods.shape == (batch_size, seq_len)
        assert distr.mu.shape == (batch_size, seq_len)
        assert distr.sigma.shape == (batch_size, seq_len)
        assert distr.nu.shape == (batch_size, seq_len)

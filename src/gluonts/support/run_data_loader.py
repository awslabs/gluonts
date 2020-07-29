import mxnet as mx
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader


def data_loader(estimator, dataset, batch):
    dataset = get_dataset(dataset)
    data = dataset.train
    epochs = 5
    batch_size = batch
    num_batches_per_epoch = 10
    bin_edges = np.array([-1e20, -1e10, 1, 1e20])
    transform = estimator.create_transformation(bin_edges=bin_edges, pred_length=dataset.metadata.prediction_length)\
        if estimator.__class__.__name__ == 'WaveNetEstimator' else estimator.create_transformation()
    loader = TrainDataLoader(
        data,
        transform=transform,
        batch_size=batch_size,
        ctx=mx.cpu(),
        num_batches_per_epoch=num_batches_per_epoch,
    )



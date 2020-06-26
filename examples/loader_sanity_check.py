import time

import torch
import mxnet as mx

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import TransformedDataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.model.deepar import DeepAREstimator

dataset = get_dataset("electricity")
dataset_train = dataset.train
freq = dataset.metadata.freq
prediction_length = dataset.metadata.prediction_length

batch_size = 32
num_batches_per_epoch = 8

estimator = DeepAREstimator(freq=freq, prediction_length=prediction_length,)

transform = estimator.create_transformation()

print("creating data loader")

training_loader = TrainDataLoader(
    dataset=dataset_train,
    transform=transform,
    batch_size=batch_size,
    ctx=mx.cpu(),
    num_batches_per_epoch=num_batches_per_epoch,
    num_workers=2,
    shuffle_buffer_length=20,
)

print("sleeping")

time.sleep(1.0)

print("done")

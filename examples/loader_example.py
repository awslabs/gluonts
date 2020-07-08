import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mxnet as mx
import torch

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.batchify import batchify as mx_batchify
from gluonts.torch.batchify import batchify as torch_batchify

dataset = get_dataset("electricity")
dataset_train = dataset.train
freq = dataset.metadata.freq
prediction_length = dataset.metadata.prediction_length

batch_size = 32
num_batches_per_epoch = 8
num_epochs = 10

estimator = DeepAREstimator(freq=freq, prediction_length=prediction_length,)

transform = estimator.create_transformation()

## What happens during training?

training_loader = TrainDataLoader(
    dataset=dataset_train,
    transform=transform,
    batch_size=batch_size,
    batchify_fn=partial(mx_batchify, ctx=mx.cpu()),
    num_batches_per_epoch=num_batches_per_epoch,
    num_workers=2,
    shuffle_buffer_length=64,
)

exp_total_batches = num_batches_per_epoch * num_epochs

start = time.time()
batch_ids = []
for epoch_no in range(num_epochs):
    for batch in training_loader:
        assert isinstance(batch["past_target"], mx.nd.NDArray)
        assert (
            batch["past_target"].shape[0] == batch_size
        ), f"{batch['past_target'].shape[0]} vs {batch_size}"
        batch_ids.append(batch["item_id"])
end = time.time()

assert (
    len(batch_ids) == exp_total_batches
), f"{len(batch_ids)} vs {exp_total_batches}"

count_per_batch = np.zeros(shape=(len(dataset_train), len(batch_ids)))
for k, ids in enumerate(batch_ids):
    for id in ids:
        count_per_batch[id, k] += 1

print(f"batches loaded: {len(batch_ids)}")
print(f"elapsed time: {end - start}")
print(f"average time: {(end - start) / len(batch_ids)}")

plt.figure(figsize=(20, 10))
sns.heatmap(count_per_batch)
plt.title("Training batches")
plt.show()

## What happens during inference?

inference_loader = InferenceDataLoader(
    dataset=dataset_train,
    transform=transform,
    batch_size=batch_size,
    batchify_fn=partial(torch_batchify, device=torch.device("cpu")),
)

start = time.time()
batch_ids = []
for batch in inference_loader:
    assert isinstance(batch["past_target"], torch.Tensor)
    assert batch["past_target"].shape[0] <= batch_size
    batch_ids.append(batch["item_id"])
end = time.time()

count_per_batch = np.zeros(shape=(len(dataset_train), len(batch_ids)))
for k, ids in enumerate(batch_ids):
    for id in ids:
        count_per_batch[id, k] += 1

print(f"batches loaded: {len(batch_ids)}")
print(f"elapsed time: {end - start}")
print(f"average time: {(end - start) / len(batch_ids)}")

plt.figure(figsize=(20, 10))
sns.heatmap(count_per_batch)
plt.title("Inference batches")
plt.show()

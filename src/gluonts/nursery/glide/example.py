import sys
from pathlib import Path

from toolz.functoolz import compose_left

from gluonts.nursery import glide
from gluonts.dataset.common import ProcessDataEntry
from gluonts.model.deepar import DeepAREstimator


from functools import partial

from dataset import JsonLinesFile


class ApplyTransformation:
    def __init__(self, t, is_train=True):
        self.t = t
        self.is_train = is_train

    def __call__(self, arg):
        # return list(self.t([arg], is_train=self.is_train))
        return self.t(arg, is_train=self.is_train)


deepar = DeepAREstimator(prediction_length=7, freq="D")
t = deepar.create_transformation()

# datasource = [{"start": "2020", "target": list(range(100))}]
datasource = JsonLinesFile(Path(sys.argv[1]))


steps = compose_left(ProcessDataEntry(freq="D"), ApplyTransformation(t),)

import time

start = time.time()

parts = glide.partition(JsonLinesFile(Path(sys.argv[1])), n=int(sys.argv[2]))


class Cycle:
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        while True:
            yield from self.it


parts = [Cycle(part) for part in parts]

# pipe = glide.ParMap(
# compose_left(ProcessDataEntry(freq="D"), ApplyTransformation(t),), parts
# )


# sum(1 for _ in pipe)

# end = time.time()
# print(end - start)


####


# class Cycle:
#     def __init__(base_iter):
#         self.base_iter = base_iter

#     def __iter__(self):
#         while True:
#             yield from self.base_iter


# def batcher(stream):


# def Dat(datasource, num_workers, transform):
#     parts = glide.partition(datasource, num_workers)

#     data = [Cycle(part) for part in parts]

#     glide.ParMap()


# train_data_loader(
#     datasource,
#     num_workers=4,
#     batch_size=128,
#     shuffle_buffer_length=10,
#     transform=[ProcessDataEntry(...), ApplyTransformation(...),],
# )

import random
import numpy as np
from functools import partial
from gluonts.dataset.parallelized_loader import stack
from gluonts.dataset.util import batcher, dct_reduce

from toolz.itertoolz import concat, take, partition
from toolz.functoolz import identity, curry


def data_loader(dataset, batch_size, buffer_size, batchify_fn):
    for superbatch in batcher(dataset, batch_size * buffer_size):
        # random.shuffle(superbatch)

        for batch in batcher(superbatch, batch_size):
            yield dct_reduce(batchify_fn, batch)


import io
import pickle
from multiprocessing.reduction import ForkingPickler


def encode(data):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(data)
    return buf.getvalue()


stack_fn = partial(stack, multi_processing=True, dtype=None)


def batchify(batch):
    return dct_reduce(stack_fn, batch)


def shuffled(xs):
    random.shuffle(xs)
    return xs


lift = curry(map)
into_batches = curry(batcher)


map_fn = compose_left(
    lift(ProcessDataEntry(freq="D")),
    ApplyTransformation(t),
    into_batches(batch_size=128 * 10),
    lift(shuffled),
    concat,
    into_batches(batch_size=128),
    lift(batchify),
)


dataset = glide.ParApply(map_fn, parts,)


for batch in take(10, dataset):
    # print(batch["past_target"].shape)
    pass


# flatten
# dataset = concat(map(pickle.loads, dataset))


# data_iter = data_loader(dataset, 128, 10, stack_fn)


# num_batches_per_epoch = 100

# sum = 0

# for epoch in range(5):
#     batches = take(num_batches_per_epoch, data_iter)

#     for batch in batches:
#         sum += 1

# print(sum)

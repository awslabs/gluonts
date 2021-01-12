import random
import sys
from functools import partial
from pathlib import Path


from toolz.itertoolz import concat
from toolz.functoolz import compose_left as c, curry

from gluonts.nursery import glide
from gluonts.nursery.datasource import FileDataset
from gluonts.nursery.datasource.schema import get_standard_schema

from gluonts.dataset.util import batcher, cycle
from gluonts.model.deepar import DeepAREstimator

ds = FileDataset(Path(sys.argv[1]))

deepar = DeepAREstimator(freq="D", prediction_length=7)

# ---- in deepar ----
schema = get_standard_schema(freq=deepar.freq)
transform = deepar.create_transformation()

batch_size = 128


batcher = curry(batcher)


def shuffled(xs):
    random.shuffle(xs)
    return xs


def create_shuffler(batch_size, batches_for_shuffling):
    return [
        batcher(batch_size=batch_size * batches_for_shuffling),
        glide.lift(shuffled),
        concat,
    ]


ds = glide.Map(schema, ds)


pipeline = [
    cycle,
    # apply schema to dataset
    # glide.lift(schema),
    # apply transformation pipeline
    partial(transform, is_train=True),
    *create_shuffler(batch_size, 10),
    partial(batcher, batch_size=batch_size),
]

train_data_loader = glide.Apply(c(*pipeline), [ds])

assert len(next(iter(train_data_loader))) == batch_size

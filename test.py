from itertools import islice

import numpy as np
from tqdm import tqdm

from gluonts.core.settings import let
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.env import env
from gluonts.itertools import Cached
from gluonts.model.deepar import DeepAREstimator

np.random.seed(0)

dataset = get_dataset("m4_daily")

estimator = DeepAREstimator(
    freq=dataset.metadata.freq,
    prediction_length=dataset.metadata.prediction_length,
    context_length=4 * dataset.metadata.prediction_length,
)

transformed_dataset = Cached(
    estimator.create_transformation().apply(dataset.train)
)

num_batches = 300_000

for batch in tqdm(
    islice(
        estimator.create_training_data_loader(transformed_dataset), num_batches
    ),
    total=num_batches,
):
    pass

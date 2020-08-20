from toolz.functoolz import compose_left

from gluonts.nursery import glide
from gluonts.dataset.common import ProcessDataEntry
from gluonts.model.deepar import DeepAREstimator


from functools import partial


class ApplyTransformation:
    def __init__(self, t, is_train=True):
        self.t = t
        self.is_train = is_train

    def __call__(self, arg):
        return list(self.t([arg], is_train=self.is_train))


deepar = DeepAREstimator(prediction_length=7, freq="D")
t = deepar.create_transformation()

data = [{"start": "2020", "target": list(range(100))}]

steps = compose_left(ProcessDataEntry(freq="D"), ApplyTransformation(t),)

pipe = glide.ParMap(steps, [data])

print(list(pipe))

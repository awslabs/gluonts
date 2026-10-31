
# Deep Learning Estimator

This RFC proposes ``DeepLearningEstimator``, a base class for deep learning
based models.


## Status Quo and Problem Statement

Despite the fact that our deep learning based models behave very similar with
respect to loading data, each estimator currently has to configure the input
processing pipeline manually. These steps are described in the following.

* Each model has an implicit schema that defines the input data. Currently, we
use `AsNumpyArray` to ensure fields are present in input data and that they
have the correct type and shape.

* Next, some features are added and the input is extended with information such
as missing value indicators.

* We then sample windows out of the processed data, using ``InstanceSplitter``
and ``InstanceSampler``.

* Windows are then grouped into batches, which are then stacked (i.e. turning
rows into columns). At last, we convert the input from ``numpy`` arrays to
framework specific types (``torch.tensor``, ``mxnet.ndarray``).

There are some complexifiers": We treat training input as an endless stream of
time series by re-iterating the training dataset over and over again. And to
increase performance we offer the option to cache the input data.
Additionally, there are three kinds of data-loaders, which all behave
differently, one each for "training", "validation" and prediction.


### Further Limitations

Both ``GluonEstimator`` and ``PyTorchLightningEstimator`` provide a common
interface to implement new estimators in MXNet and PyTorch respectively.

However, both classes only provide a transformation chain to handle input data,
meaning that these pipelines need to handle schema validation as well as
feature processing.

Since input time series are represented using plain dictionaries, code handling
these need a lot of additional information to be able to work on them. Thus,
each estimator currently defines its own ``InstanceSplitter`` and configures
dataloading.


## Proposal

We can simplify the implementation of estimators using a common way to handle
data loading.

We replace our dictionary based approach with  ``zebras.TimeFrame`` and
``zebras.SplitFrame`` to handle time series data. This has two advantages:
We can use ``zebras.Schema`` to specify input data for a given estimator; and
implementing transformation steps become a lot easier.

We introduce ``DeepLearningInterface`` which provides a more granular interface
to load data. It requires each derived estimator to provide information such
as ``past_length`` which is used to construct batches of the correct size.
Using that information it then can provide default implementations for methods
such as ``.training_instance_splitter(...)`` .

To further simplify configuration we move some options to a ``Settings``
object (dependency injection). This has the advantage that we can alter some
training behaviour without altering the estimator code. For example, we would
configure data caching through settings instead of passing these arguments to
the estimator or train methods:

```py
with setting._let(cache_data=True):
    estimator.train(...)
```

## Implementation

We introduce a new class ``DeepLearningEstimator``:

```py

class DeepLearningEstimator:
    def get_schema(self) -> zb.Schema:
        raise NotImplementedError

    def training_pipeline(self):
        return []

    def prediction_pipeline(self):
        self.training_pipeline()

    def train_model(self, training_data, validation_data):
        raise NotImplementedError

    # default implementations

    def train(self, training_data, validation_data=None):
        training_data = self.training_dataloader(training_data)
        validation_data = maybe.map(
            validation_data, self.validation_dataloader
        )

        return self.train_model(training_data, validation_data)

    def training_instance_splitter(self):
        ...

    def training_dataloader(self, training_data):
        ...

    def validation_dataloader(self, validation_data):
        ...

```

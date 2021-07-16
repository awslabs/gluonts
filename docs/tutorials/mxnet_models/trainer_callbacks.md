# Trainer callbacks

This notebook illustrates how one can control the training procedure of MXNet-based models by providing callbacks to the `Trainer` class.
A callback is a function which gets called at one or more specific hook points during training.
You can use predefined GluonTS callbacks like `TrainingHistory`, `ModelAveraging` or `TerminateOnNaN`, or you can implement your own callback.

```python
from gluonts.dataset.repository.datasets import get_dataset

dataset = "m4_hourly"
dataset = get_dataset(dataset)
prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
```

## Using a single callback

To use callbacks, simply pass them as a list when constructing the `Trainer`:
in the following example, we are using the `TrainingHistory` callback to record loss values measured during training.

```python
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import TrainingHistory

# defining a callback, which will log the training loss for each epoch
history = TrainingHistory()

trainer = Trainer(epochs=3, callbacks=[history])
estimator = SimpleFeedForwardEstimator(prediction_length=prediction_length, freq=freq, trainer=trainer)

predictor = estimator.train(dataset.train, num_workers=None)
```

Print the training loss over the epochs:

```python
print(history.loss_history)
```

## Using multiple callbacks

To continue the training from a given predictor you can use the `WarmStart` callback.
When you want to use more than one callback, just provide a list with multiple callback objects:

```python
from gluonts.mx.trainer.callback import WarmStart

warm_start = WarmStart(predictor=predictor)

trainer=Trainer(epochs=3, callbacks=[history, warm_start])

estimator = SimpleFeedForwardEstimator(prediction_length=prediction_length, freq=freq, trainer=trainer)

predictor = estimator.train(dataset.train, num_workers=None)
```

```python
print(history.loss_history) # The training loss history of all 3+3 epochs we trained the model for
```

## Default callbacks

In addition to the callbacks you specify, the `Trainer` class uses the two default callbacks `ModelAveraging` and `LearningRateReduction`.
You can turn them off by setting `add_default_callbacks=False` when initializing the Trainer.

```python
trainer=Trainer(epochs=20, callbacks=[history]) # use the TrainingHistory Callback and the default callbacks.
trainer=Trainer(epochs=20, callbacks=[history], add_default_callbacks=False) # use only the TrainingHistory Callback
trainer=Trainer(epochs=20, add_default_callbacks=False) # use no callback at all
```

## Custom callbacks

To implement your own callback you can write a class which inherits from `gluonts.mx.trainer.Callback`, and overwrite one or more of the hooks.
Have a look at the abstract `Callback` class, the hooks take different arguments which you can use. 
Hook methods with boolean return value stop the training if False is returned.

Here is an example for a custom callback implementation which terminates training early based on the value of some metric (such as the RMSE).
It only implements the hook method `on_epoch_end` which gets called after all batches of one epoch have been processed.

```python
import numpy as np
import mxnet as mx

from gluonts.evaluation import Evaluator
from gluonts.dataset.common import Dataset
from gluonts.mx import copy_parameters, GluonPredictor
from gluonts.mx.trainer.callback import Callback


class MetricInferenceEarlyStopping(Callback):
    """
    Early Stopping mechanism based on the prediction network.
    Can be used to base the Early Stopping directly on a metric of interest, instead of on the training/validation loss.
    In the same way as test datasets are used during model evaluation,
    the time series of the validation_dataset can overlap with the train dataset time series,
    except for a prediction_length part at the end of each time series.

    Parameters
    ----------
    validation_dataset
        An out-of-sample dataset which is used to monitor metrics
    predictor
        A gluon predictor, with a prediction network that matches the training network
    evaluator
        The Evaluator used to calculate the validation metrics.
    metric
        The metric on which to base the early stopping on.
    patience
        Number of epochs to train on given the metric did not improve more than min_delta.
    min_delta
        Minimum change in the monitored metric counting as an improvement
    verbose
        Controls, if the validation metric is printed after each epoch.
    minimize_metric
        The metric objective.
    restore_best_network
        Controls, if the best model, as assessed by the validation metrics is restored after training.
    num_samples
        The amount of samples drawn to calculate the inference metrics.
    """

    def __init__(
        self,
        validation_dataset: Dataset,
        predictor: GluonPredictor,
        evaluator: Evaluator = Evaluator(num_workers=None),
        metric: str = "MSE",
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True,
        minimize_metric: bool = True,
        restore_best_network: bool = True,
        num_samples: int = 100,
    ):
        assert (
            patience >= 0
        ), "EarlyStopping Callback patience needs to be >= 0"
        assert (
            min_delta >= 0
        ), "EarlyStopping Callback min_delta needs to be >= 0.0"
        assert (
            num_samples >= 1
        ), "EarlyStopping Callback num_samples needs to be >= 1"

        self.validation_dataset = list(validation_dataset)
        self.predictor = predictor
        self.evaluator = evaluator
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_network = restore_best_network
        self.num_samples = num_samples

        if minimize_metric:
            self.best_metric_value = np.inf
            self.is_better = np.less
        else:
            self.best_metric_value = -np.inf
            self.is_better = np.greater

        self.validation_metric_history: List[float] = []
        self.best_network = None
        self.n_stale_epochs = 0

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: mx.gluon.nn.HybridBlock,
        trainer: mx.gluon.Trainer,
        best_epoch_info: dict,
        ctx: mx.Context
    ) -> bool:
        should_continue = True
        copy_parameters(training_network, self.predictor.prediction_net)

        from gluonts.evaluation.backtest import make_evaluation_predictions

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.validation_dataset,
            predictor=self.predictor,
            num_samples=self.num_samples,
        )

        agg_metrics, item_metrics = self.evaluator(ts_it, forecast_it)
        current_metric_value = agg_metrics[self.metric]
        self.validation_metric_history.append(current_metric_value)

        if self.verbose:
            print(
                f"Validation metric {self.metric}: {current_metric_value}, best: {self.best_metric_value}"
            )

        if self.is_better(current_metric_value, self.best_metric_value):
            self.best_metric_value = current_metric_value

            if self.restore_best_network:
                training_network.save_parameters("best_network.params")

            self.n_stale_epochs = 0
        else:
            self.n_stale_epochs += 1
            if self.n_stale_epochs == self.patience:
                should_continue = False
                print(
                    f"EarlyStopping callback initiated stop of training at epoch {epoch_no}."
                )

                if self.restore_best_network:
                    print(
                        f"Restoring best network from epoch {epoch_no - self.patience}."
                    )
                    training_network.load_parameters("best_network.params")

        return should_continue
```

We can now use the custom callback as follows.
Note that we're running an extremely short number of epochs, simply to keep the runtime of the notebook manageable:
feel free to increase the number of epochs to properly test the effectiveness of the callback.

```python
estimator = SimpleFeedForwardEstimator(prediction_length=prediction_length, freq=freq)
training_network = estimator.create_training_network()
transformation = estimator.create_transformation()

predictor = estimator.create_predictor(transformation=transformation, trained_network=training_network)

es_callback = MetricInferenceEarlyStopping(validation_dataset=dataset.test, predictor=predictor, metric="MSE")

trainer = Trainer(epochs=5, callbacks=[es_callback])

estimator.trainer = trainer

pred = estimator.train(dataset.train)
```
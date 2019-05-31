
# Time series forecasting with GluonTS

The GluonTS toolkit contains components and tools for building time series models using MXNet. The models that are currently included are forecasting models but the components also support other time series use cases, such as classification or anomly detection. 

The toolkit is not intended as a forecasting solution for businesses or end users but it rather targets scientists and engineers who want to tweak algorithms or build and experiment with their own models.  

GluonTS contains:
- Components for building new models (likelihoods, feature processing pipelines, calendar features etc.)
- Data loading and processing
- A number of pre-built models
- Plotting and evaluation facilities
- Artificial and real datasets (only external datasets with blessed license)


```python
%matplotlib inline
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from itertools import islice
from pathlib import Path
```

## Loading datasets

GluonTS comes with a number of publicly available datasets. 


```python
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
```


```python
print(f"Available datasets: {list(dataset_recipes.keys())}")
```

An available dataset can be easily downloaded by its name. If the dataset already exists locally it is not downloaded again by setting `regenerate=False`.


```python
dataset = get_dataset("m4_hourly", regenerate=False)
```

### What is in a dataset?

In general, the datasets provided by GluonTS are objects that consists of three main components:
- `dataset.train` is an iterable collection of data entries used for training.
- `dataset.test` is an iterable collection of data entries used for inference. The test dataset is an extended version of the train dataset that contains a window in the end of each time series that was not seen during training. This window has length equal to the recommended prediction length.
- `dataset.metadata` containts metadata of the dataset such as the frequency of the time series, a recommended prediction horizon, associated features, etc. 

Of course a custom dataset does not need to have this specific format but it should be iterable and it should have a "target" and a "start" field.


```python
entry = next(iter(dataset.train))
train_series = to_pandas(entry)
train_series.plot()
plt.show()
```


```python
entry = next(iter(dataset.test))
test_series = to_pandas(entry)
test_series.plot()
plt.axvline(train_series.index[-1], color='r') # end of train dataset
plt.show()
```


```python
print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")
print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
print(f"Frequency of the time series: {dataset.metadata.time_granularity}")
```

## Training an existing model (`Estimator`)

As we already mentioned, GluonTS comes with a number of pre-built models that can be used directly with minor hyperparameter configurations. For starters we will use one of these predefined models to go through the whole pipeline of training a model, predicting, and evaluating the results. 

GluonTS gives focus (but is not restricted) to probabilistic forecasting, i.e., forecasting the future distribution of the values and not the future values themselves (point estimates) of a time series. Having estimated the future distribution of each time step in the forecasting horizon, we can draw a sample from the distribution at each time step and thus create a "sample path", that can be seen as a possible realization of the future. In practice we draw multiple samples and create multiple sample paths which can be used for visualization, evalution of the model, to derive statistics, etc.

In this example we will use a simple pre-built feedforward neural network estimator that takes as input a window of length `context_length` and predicts the distribution of the values of the subsequent future window of length `prediction_length`. 

In general, each estimator (pre-built or custom) is configured by a number of hyperparameters that can be either common (but not binding) among all estimators (e.g. the `prediction_length`) or specific for the particular estimator (e.g. number of layers for a neural network or the stride in a CNN).

Finally, each estimator is configured by a `Trainer`, which defines how the model will be trained. i.e., the number of epochs, the learning rate, etc.  


```python
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
```


```python
estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=100,
    freq=dataset.metadata.time_granularity,
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=200,),
)
```

After specifing our estimator with all the necessary hyperparameters we can train it using our training dataset `dataset.train` just by invoking the `.train` method of the estimator. The training returns a predictor that can be used to predict.


```python
predictor = estimator.train(dataset.train)
```

Now we have a predictor in our hands. We can use it to predict the last window of the `dataset.test` and evaluate how our model performs.

GluonTS comes with the `make_evaluation_predictions` function that automates all this procedure. Roughly, this module performs the following steps:
- Removes the final window of length `prediction_length` of the `dataset.test` that we want to predict
- The estimator uses the remaining dataset to predict (in the form of sample paths) the "future" window that was just removed 
- The module outputs a generator over the forecasted sample paths and a generator over the `dataset.test` 


```python
from gluonts.evaluation.backtest import make_evaluation_predictions
```


```python
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,  # test dataset
    predictor=predictor,  # predictor
    num_eval_samples=100,  # number of sample paths
)
```


```python
print(type(forecast_it))
print(type(ts_it))
```

First, we can convert these generators to lists to ease the subsequent computations.


```python
forecasts = list(forecast_it)
tss = list(ts_it)
```

Now, let's see what do these lists contain under the hood. Let's start with the time series `tss` that is simpler. Each item in the `tss` list is just a pandas dataframe that contains the actual time series.


```python
print(type(tss[0]))
```


```python
k = 3
tss[0].head(k)  # show the k first values
```

The `forecasts` list is a bit more complex. Each item in the `forecasts` list is an object that contains all the sample paths, the start date of the forecast, the frequency of the time series, etc. We can access all these information by simply invoking the corresponding attribute of the forecast object.


```python
print(type(forecasts[0]))
```


```python
print(f"Number of sample paths: {forecasts[0].num_samples}")
print(f"Start date of the forecast window: {forecasts[0].start_date}")
print(f"Frequency of the time series: {forecasts[0].freq}")
```

Apart from retrieving basic information we can do some more complex calculation such as to compute the mean or a given quantile of the values of the forecasted window.


```python
print(f"Mean of the future window:\n {forecasts[0].mean}")
print(f"0.5-quantile (median) of the future window:\n {forecasts[0].quantile(0.5)}")
```

Finally, each forecast object has a `.plot` method that can be parametrized to show the mean, prediction intervals, etc.


```python
plot_length = 150 
tss[0][-plot_length:].plot()  # plot the time series
forecasts[0].plot(prediction_intervals=(50.0, 80.0, 95.0), # confidence intervals (plotted in different shades)
              show_mean=True,
              color='g')
plt.grid()
plt.show()
```

We can also evaluate the quality of our forecasts. The `Evaluator` returns aggregate error metrics as well as metric per time series which can be used e.g. for scatter plots.


```python
from gluonts.evaluation import Evaluator
```


```python
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))
```


```python
print(json.dumps(agg_metrics, indent=4))
```


```python
item_metrics.head(10)
```


```python
item_metrics.plot(x='abs_target_mean', y='MASE', kind='scatter')
plt.plot()
```

## Create your own forecast model

For creating your own forecast model you need to:
- define the training and prediction network
- define a new estimator that specifies any data processing and uses the networks

The training and prediction networks can be arbitrarily complex but they should follow some basic rules:
- Both should have a `hybrid_forward` method that defines what should happeni when the network is called    
- The trainng network's `hybrid_forward` should return a **loss** based on the prediction and the true values
- The prediction network's `hybrid_forward` should return the predictions 

For example, we can create a simple training network that defines a neural network with output dimension equal to `prediction_length` and uses the L1 loss in the `hybrid_forward` method, and a prediction network that is (and should be) identical to the training network (by inheriting the class).


```python
class MyTrainNetwork(gluon.HybridBlock):
    def __init__(self, prediction_length, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
    
        with self.name_scope():
            # Set up a 3 layer neural network that directly predicts the target values
            self.nn = mx.gluon.nn.HybridSequential()
            self.nn.add(mx.gluon.nn.Dense(units=40, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=40, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length, activation='softrelu'))

    def hybrid_forward(self, F, past_target, future_target):
        prediction = self.nn(past_target)
        # calculate L1 loss with the future_target to learn the median
        return (prediction - future_target).abs().mean(axis=-1)


class MyPredNetwork(MyTrainNetwork):
    # The prediction network only receives past_target and returns predictions
    def hybrid_forward(self, F, past_target):
        prediction = self.nn(past_target)
        return prediction.expand_dims(axis=1)
```

Now, we need to construct the estimator which should also follow some rules:
- It should include a `create_transformation` method that defines all the possible feature transformations and how the data is split during training
- It should include a `create_training_network` method that returns the training network configured with any necessary hyperparameters
- It should include a `create_predictor` method that creates the prediction network, and returns a `Predictor` object 

A `Predictor` defines the `predictor.predict` method of a given predictor. This method takes the test dataset, it passes it through the prediction network to take the predictions, and yields the predictions. You can think of the `Predictor` object as a wrapper of the prediction network that defines its `predict` method. 

Earlier, we used the `make_evaluation_predictions` to evaluate our predictor. Internally, the `make_evaluation_predictions` function invokes the `predict` method of the predictor to get the forecasts.


```python
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.core.component import validated
from gluonts.support.util import copy_parameters
from gluonts.transform import ExpectedNumInstanceSampler, Transformation, InstanceSplitter, FieldName
from mxnet.gluon import HybridBlock
```


```python
class MyEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        trainer: Trainer = Trainer()
    ) -> None:
        super().__init__(trainer=trainer)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        
    
    def create_transformation(self):
        # Feature transformation that the model uses for input. 
        # Here we use a transformation that randomly select training samples from all time series.
        return InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                )
    
    def create_training_network(self) -> MyTrainNetwork:
        return MyTrainNetwork(
            prediction_length=self.prediction_length
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = MyPredNetwork(
            prediction_length=self.prediction_length
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
```

Now, we can repeat the same pipeline as in the case we had a pre-built model: train the predictor, create the forecasts and evaluate the results.


```python
estimator = MyEstimator(
    prediction_length=dataset.metadata.prediction_length,
    context_length=200,
    freq=dataset.metadata.time_granularity,
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=200,),
)
```


```python
predictor = estimator.train(dataset.train)
```


```python
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test, 
    predictor=predictor, 
    num_eval_samples=100
)
```


```python
forecasts = list(forecast_it)
tss = list(ts_it)
```


```python
plot_length = 150 
tss[0][-plot_length:].plot()  # plot the time series
forecasts[0].plot(prediction_intervals=(50.0, 80.0, 95.0), # confidence intervals (plotted in different shades)
              show_mean=True,
              color='g')
plt.grid()
plt.show()
```

We observe from the plot above that we cannot actually see any confidence interval in the predictions. This is expected since the networks that we defined do not do probabilistic forecasting but they just give point estimates. That is, the prediction netwrok does not return samples from a learned distribution. Rather, for a fixed input the network gives always the same output. So, by requiring 100 sample paths in such a network, we get 100 times the same output.


```python
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))
```


```python
print(json.dumps(agg_metrics, indent=4))
```


```python
item_metrics.head(10)
```


```python
item_metrics.plot(x='abs_target_mean', y='MASE', kind='scatter')
plt.plot()
```

```{admonition} **Draft**

This article may be extended or reworked in the future.

```

# Background

## What is Time Series Forecasting?

Generally speaking, forecasting just means making predictions about events in
the future. Trivially, in time series forecasting we want to predict the
future values of a given time series.

For example, in electricity production it is very important that demand and
supply are in balance. Thus, producers anticipate consumer demand for
electricity and plan production capacity accordingly. In other words, producers
rely on accurate time series forecasting of consumer demand for electricity to
generate just enough supply.

In forecasting, there is the implicit assumption that observable behaviours of
the past that impact time series values continue into the future. To stay
with the electricity example: People will generally consume less energy in the
night than during the day, will watch TV mostly during the evenings and use
air conditioners when it's hot during summer.

```{figure} ../_static/electricity-10w.png
---
---
Ten weeks of data plotted over each other -- ``electricity`` dataset.
```

Naturally, it's impossible to forecast the unpredictable. For instance, in 2019
it was virtually impossible to account for the possibility of travel
restrictions due to the Covid-19 pandemic when trying to forecacst travel demand
for 2020.

Thus, forecasting operates on the caveat that the underlying factors that
generate the time series values don't fundamentally change in the future. It is
a tool to predict the ordinary and not the surprising.

To look at this another way: Models are actually trained to predict the past
and it is only us who use models to forecast into the future.


## Target And Features

We call the time series that we want to predict the `target` time series. The
past target values are the most important information a model can use to make
accurate predictions.

In addition, models can make use of features, additional values that have an
impact on the target value. We differentiate between "static" and "dynamic"
features.

A dynamic feature can be different for every time point. For example, this
could be the price of a product, but also more general information such as
outside air temperature. Internally, we generate dynamic features for things
like the age of the time series or what day of the week it is.

```{important}

Most models require dynamic features to be available in the future time range
when making predictions.

```

In contrast, static features describe a time series independently of time. If
we were to predict different products across different stores, we can use
static features to label each time series to include store and product
indentifiers.

We further differentiate between categorical and continuous (real) features. The
idea is that in continuous features the number itself has meaning, for example
when using the price as a feature. A categorical feature on the other hand
doesn't have the same property: Stores `0`, `1`, and `2` are distinct entities
and there is no notion of having a "higher store".

<!-- TODO: Have some nice example examplifying the above. -->

<!-- ```{admonition} Example -->

<!-- Image we are the owner of a cafe. -->

<!-- ``` -->


## Probabilistic Forecasting

One core idea in GluonTS is that we don't produce simple values as forecasts,
but actually predict distributions.

An intuitive way to look at this is to imagine predicting a time series 100
times, which returns 100 different time series samples, which form a
distribution around them - except that we can directly emit these distributions
and then draw samples from them.

Distributions provide the benefit that they provide a range of likely values.
Imagine being a restaurant owner, wondering how many ingredients to buy; if we
buy too few we won't serve customer demand, but buying too many will produce
waste. Thus, when we forecast demand, it is valuable if a model can tell us
that there is probably a demand of say 50 dishes, but unlikely more than 60.

```{figure} ../_static/forecast-distributions.png
---
---
Predicting 24 hours, showing `p50`, `p90`, `p95`, `p98` prediction intervals.
```

```{note}

The predicted distributions are not authorative: A predicted 90th percentile
doesn't mean that only 10% of actual values will be of higher value, but that
this is the guess of the model about where this line is.

```

## Local and Global Models

In GluonTS we use the concepts of local and global models.

A local model is fit for a single time series and used to make predictions for
that time series, whilst a global model is trained across many time series and
a single global model is used to make predictions for all time series of a
dataset.

Training a global model can take a lot of time: up to hours, but sometimes even
days. Thus, it is not feasible to train the model as part of the prediction
request and it happens as a separate "offline" step. In contrast, fitting a
local model is usually much faster and is done "online" as part of the
prediction.

In GluonTS, local models are directly available as predictors, whilst global
models are offered as estimators, which need to be trained first.


<!-- TODO -->
<!-- ## Train Test Split -->
<!-- ## Measuring Accuracy -->

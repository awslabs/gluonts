# Synthetic Data Generation Tutorial

```python
import json
from itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
```


```python
from gluonts.dataset.artificial import recipe as rcp
from gluonts.core.serde import dump_code, load_code
```


```python
# plotting utils

def plot_recipe(recipe, length):
    output_dict = rcp.evaluate(recipe, length)
    K = len(output_dict)
    lct = MultipleLocator(288)
    minor = AutoMinorLocator(12)

    fig, axs = plt.subplots(K, 1, figsize=(16, 2 * len(recipe)))
    for i, k in enumerate(output_dict):
        axs[i].xaxis.set_major_locator(lct)
        axs[i].xaxis.set_minor_locator(minor)
        axs[i].plot(output_dict[k])
        axs[i].grid()
        axs[i].set_ylabel(k)


def plot_examples(recipe, target, length, num, anomaly_indicator=None):
    fix, axs = plt.subplots(num, 1, figsize=(16, num * 2))
    for i in range(num):
        xx = rcp.evaluate(recipe, length)
        axs[i].plot(xx[target])
        axs[i].set_ylim(0, 1.1*np.max(xx[target]))
        axs[i].grid()
        if anomaly_indicator is not None:
            axs[i].fill_between(
                np.arange(len(xx[target])), 
                xx[anomaly_indicator] * 1.1*np.max(xx[target]), 
                np.zeros(len(xx[target])), 
                alpha=0.3,
                color="red")


def print_dicts(*dicts):
    for d in dicts:
        print("{")
        for k,v in d.items():
            print("\t", k, ": ", v)
        print("}\n")
```

## Recipes

Recipes are lists of `(name, expression)` tuples. The role of a recipe is to describe the generative process of a single time series. In order to do so, the `expression`s in the `(name, expression)` pairs are evaluated for each time series in the order given in the list to produce a `{name: value}` dictionary as output.


```python
recipe = [
    ("myOutput1", 1.0),
    ("myOutput2", 42)
]

rcp.evaluate(recipe, length=5)
```

### Expressions

Each `expression` can either be a Python value, a string (interpreted as a reference to previously defined `name`), or a special type of `Callable`, that is evaluated each time the recipe is evaluated. 


```python
recipe = [
    ("myOutput1", 1.0),
    ("myOutput2", "myOutput1")  # reference to previously defined name
]

rcp.evaluate(recipe, length=5)
```


```python
recipe = [
    ("myOutput1", rcp.RandomGaussian()),  # callable as expression
]

# multiple evaluations lead to different results, due to randomness
print_dicts(
    rcp.evaluate(recipe, length=5),
    rcp.evaluate(recipe, length=5),
)
```

### Expressions with References


```python
recipe = [
    ("stddev1", 2.0),
    ("stddev2", rcp.RandomUniform(low=0, high=1, shape=(1, ))),
    ("myOutput1", rcp.RandomGaussian(stddev="stddev1")),
    ("myOutput2", rcp.RandomGaussian(stddev="stddev2"))
]

# multiple evaluations lead to different results, due to randomness
print_dicts(
    rcp.evaluate(recipe, length=5),
    rcp.evaluate(recipe, length=5)
)
```


```python
recipe = [
    ("random_out", rcp.RandomGaussian(shape=(1,))),
    ("fixed_out", np.random.randn(1))
]

# note that fixed_out stays the same; 
# it's evaluated only once when the recipe is created
print_dicts(
    rcp.evaluate(recipe, length=1),
    rcp.evaluate(recipe, length=1)
)
```

### Length

Most operators in the `recipe` package have a `length` argument that is automatically passed when the expression is evaluated. The idea is that these recipes are used to generate fixed-length time series, and most operators produce
individual components of the time series that have the same length. 


```python
recipe = [
    ("random_gaussian", rcp.RandomGaussian()),
    ("constant_vec", rcp.ConstantVec(42))
]
     
print_dicts(
    rcp.evaluate(recipe, length=3),
    rcp.evaluate(recipe, length=5)
)
```

### Operator Overloading

The `Callable` operators defined in the `recipe` package overload the basic arithmetic operations (addition, subtraction, multiplication, division).


```python
recipe = [
    ("x1", 42 * rcp.ConstantVec(1)),
    ("x2", "x1" * rcp.RandomUniform()),
    ("x3", rcp.RandomGaussian() + rcp.RandomUniform()), 
    ("x4", rcp.Ref("x1") + "x2" + "x3")
]

rcp.evaluate(recipe, 3)
```

### SerDe

Recipes composed of serializable / representable components can easily be serialized / deserialized.


```python
dumped = dump_code(recipe)
print(dumped)

reconstructed = load_code(dumped)

rcp.evaluate(reconstructed, 3)
```

## Simple Examples


```python
recipe = [
    ("daily_smooth_seasonality", rcp.SmoothSeasonality(period=288, phase=-72)),
    ("noise", rcp.RandomGaussian(stddev=0.1)),
    ("signal", rcp.Add(["daily_smooth_seasonality", "noise"]))
]

plot_recipe(recipe, 3 * 288)

```


```python
recipe = [
    ("slope", rcp.RandomUniform(low=0, high=3, shape=(1,))),
    ("trend", rcp.LinearTrend(slope="slope")),
    ("daily_smooth_seasonality", rcp.SmoothSeasonality(period=288, phase=-72)),
    ("noise", rcp.RandomGaussian(stddev=0.1)),
    ("signal", rcp.Add(["trend", "daily_smooth_seasonality", "noise"]))
]

plot_examples(recipe, "signal", 3 * 288, 5)
```


```python
weekly_seasonal_unscaled = [
    ('daily_smooth_seasonality', rcp.SmoothSeasonality(period=288, phase=-72)),
    ('weekday_scale', rcp.RandomUniform(0.1, 10, shape=(1,))),
    ('weekly_pattern', rcp.NormalizeMax(rcp.Concatenate([rcp.Ref("weekday_scale") * np.ones(5), np.ones(2)]))),
    ('day_of_week', rcp.Dilated(rcp.Repeated('weekly_pattern'), 288)),
    ('level', rcp.RandomUniform(low=0, high=10, shape=1)),
    ('noise_level', rcp.RandomUniform(low=0.01, high=1, shape=1)),
    ('noise', 'noise_level' * rcp.RandomGaussian()),
    ('signal', rcp.Mul(['daily_smooth_seasonality','day_of_week'])),
    ('unscaled', rcp.Add(['level', 'signal', 'noise']))
]
     
plot_recipe(weekly_seasonal_unscaled, 10 * 288)
plot_examples(weekly_seasonal_unscaled, "unscaled", 10 * 288, 5)
```

## Composing Recipes

As recipes are just lists of expressions that evaluated sequentially, recipes can simply be composed from smaller component recipes by concatenating the corresponding lists. It is also possible to include the output of one recipe inside another one using the `EvalRecipe` operator.


```python
scaling = [
    ("scale", rcp.RandomUniform(0, 1000)),
    ("z", "scale" * rcp.Ref("unscaled"))
]

weekly_seasonal = weekly_seasonal_unscaled + scaling

plot_examples(weekly_seasonal, "z", 10 * 288, 5)
```


```python
weekly_seasonality = [
    ('daily_pattern', rcp.RandomUniform(0, 1, shape=(24,))),
    ('daily_seasonality', rcp.Dilated(rcp.Repeated("daily_pattern"), 12)),
    ('weekly_pattern', rcp.RandomUniform(0, 1, shape=(7,))),
    ('weekly_seasonality', rcp.Dilated(rcp.Repeated("weekly_pattern"), 288)),
    ('unnormalized_seasonality', rcp.Mul(['daily_seasonality', 'weekly_seasonality'])),
    ('seasonality', rcp.NormalizeMax("unnormalized_seasonality")),
]

gaussian_noise_low = [
    ('noise_level', rcp.RandomUniform(low=0.01, high=0.1, shape=1)),
    ('noise', rcp.Ref('noise_level') * rcp.RandomGaussian()),
]

complex_weekly_seasonal = (
      weekly_seasonality 
    + [
        ('level', rcp.RandomUniform(low=0, high=10, shape=1)),
        ('signal', rcp.Add(['level', 'seasonality']))
    ]
    + gaussian_noise_low
    + [("unscaled", rcp.Add(["signal", "noise"]))]
    + scaling
)

plot_examples(complex_weekly_seasonal, "z", 10 * 288, 5)
```

## Generating Anomalies

Anomalies are just another effect added/multiplied to a base time series. We can define a recipe for creating certain types of anomalies, and then compose it with a base recipe.


```python
constant_recipe = [
    ("z", rcp.ConstantVec(1.0))
]

bmc_scale_anomalies = [
    ('normal_indicator', rcp.BinaryMarkovChain(one_to_zero=1/(288*7), zero_to_one=0.1)),
    ('anomaly_indicator', rcp.OneMinus('normal_indicator')),
    ('anomaly_scale', 0.5 + rcp.RandomUniform(-1.0, 1.0, shape=1)),
    ('anomaly_multiplier', 1 + rcp.Ref('anomaly_scale') * rcp.Ref('anomaly_indicator')),
    ('target', rcp.Mul(['z', 'anomaly_multiplier']))
]

plot_examples(constant_recipe + bmc_scale_anomalies, "target", 10*288, 5, "anomaly_indicator")
```


```python
plot_examples(weekly_seasonal + bmc_scale_anomalies, 'target', 288*7, 5, "anomaly_indicator")
```

## Generating Changepoints


```python
homoskedastic_gaussian_noise = [
    ('level', rcp.RandomUniform(0, 10, shape=1)),
    ('noise_level', rcp.RandomUniform(0.01, 1, shape=1)),
    ('noise', rcp.RandomGaussian("noise_level")),
    ('unscaled', rcp.Add(['level', 'noise'])), 
]
```


```python
changepoint_noise_to_seasonal = [
    ('z_1', rcp.EvalRecipe(homoskedastic_gaussian_noise, "unscaled")), 
    ('z_2', rcp.EvalRecipe(weekly_seasonal_unscaled, "unscaled")),
    ('z_stacked', rcp.StackPrefix('z')),
    ('change', rcp.RandomChangepoints(1)),
    ('unscaled', rcp.Choose("z_stacked", "change"))
]

changepoint_noise_to_seasonal_scaled = changepoint_noise_to_seasonal + scaling
```


```python
plot_examples(changepoint_noise_to_seasonal_scaled + bmc_scale_anomalies, 'target', 288*7, 10, "anomaly_indicator")
```

## Generating several time series


```python
rcp.take_as_list(rcp.generate(10, weekly_seasonal_unscaled, "2018-01-01", {}), 2)
```

## Saving to a file


```python
def write_to_file(recipe, length, num_ts, fields, fn):
    with open(fn, 'w') as f, open(fn+"-all", 'w') as g:
        for x in islice(rcp.generate(length, recipe, "2019-01-07 00:00"), num_ts):
            z = {}
            for k in x:
                if type(x[k]) == np.ndarray:
                    z[k] = x[k].tolist()
                else:
                    z[k] = x[k]
            xx = {}
            for fi in fields:
                xx[fi] = z[fi]
            try:
                f.write(json.dumps(xx))
            except Exception as e:
                print(xx)
                print(z)
                raise e
            f.write('\n')
            g.write(json.dumps(z))
            g.write('\n')
```

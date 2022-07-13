# Data Splitter Usage

In this notebook, we are going to show how to use the `split` method existed in our GluonTS project.

In the `split` method:
- you need to provide the `dataset` for the method to split.
- you also need to provide `offset` or `date`, but not both simultaneously. These two arguments are provided for the method to know how to slices training and test data based on a fixed integer offset or a ``pandas.Period``.
As a result, the `split` method returns the splited dataset, consisting of the training data `training_dataset` and the TestTemplate objectives `test_template` which knows how to generate test data `test_pairs` using the memeber function `generate_instances`.


## Data loading and processing


```python
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.split.splitter import OffsetSplitter, DateSplitter, split
from gluonts.dataset.util import to_pandas
```


```python
%matplotlib inline
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
```

### Provided datasets


```python
url = u"https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(url, header=0)
whole_dataset = PandasDataset(df, timestamp="timestamp", target="value")
```


```python
plt.figure(figsize=(30, 8))
to_pandas(list(whole_dataset)[0]).plot(color='royalblue')
plt.grid(which="both")
plt.legend(["whole dataset"], loc="upper left")
plt.show()
```

### Aggregate and get smaller datasets


```python
df["timestamp"] = pd.to_datetime(df["timestamp"])
df2 = df.set_index("timestamp").resample("1H").sum().reset_index()
sub_dataset = PandasDataset(df2, timestamp="timestamp", target="value")
```


```python
plt.figure(figsize=(20, 3))
to_pandas(list(sub_dataset)[0]).plot(color='royalblue')
plt.grid(which="both")
plt.legend(["sub dataset"], loc="upper left")
plt.show()
```

## Data split

we will take training data up to date `2015-04-07 00:00:00`, then generate several test instances from there onwards


```python
def get_start_end(entry):
    start = entry['start']
    end = entry['start'] + len(entry['target']) * entry['start'].freq
    return start, end
```


```python
date = pd.Period("2015-04-07 00:00:00", freq='1H')
prediction_length=3 * 24
training_dataset, test_template = split(sub_dataset, date=date)
test_pairs = test_template.generate_instances(
                prediction_length=prediction_length,
                windows=3,
             )
```


```python
for original_entry, train_entry in zip(sub_dataset, training_dataset):
    start, end = get_start_end(train_entry)
    plt.figure(figsize=(20,3))
    to_pandas(original_entry).plot(color='royalblue')
    plt.grid(which="both")
    plt.axvspan(start, end, facecolor='red', alpha=.2)
    plt.legend(["sub dataset"], loc="upper left")

for original_entry in sub_dataset:
    for test_input, test_label in test_pairs:
        start_input, end_input = get_start_end(test_input)
        start_label, end_label = get_start_end(test_label)
        plt.figure(figsize=(20,3))
        to_pandas(original_entry).plot(color='royalblue')
        plt.grid(which="both")
        plt.axvspan(start_input, end_input, facecolor='green', alpha=.2)
        plt.axvspan(start_label, end_label, facecolor='blue', alpha=.2)
        plt.legend(["sub dataset"], loc="upper left")
```

we will take training data up to date `2015-03-27 00:00:00`, then generate several test instances from date `2015-04-07 00:00:00` onwards


```python
TRAINING_END_DATE = pd.Period("2015-03-27 00:00:00", freq='1H')
TEST_START_DATE = pd.Period("2015-04-07 00:00:00", freq='1H')
training_dataset, _ = split(sub_dataset, date=TRAINING_END_DATE)
_, test_template = split(sub_dataset, date=TEST_START_DATE)
test_pairs = test_template.generate_instances(
                prediction_length=prediction_length,
                windows=3,
             )
```


```python
for original_entry, train_entry in zip(sub_dataset, training_dataset):
    start, end = get_start_end(train_entry)
    plt.figure(figsize=(20,3))
    to_pandas(original_entry).plot(color='royalblue')
    plt.grid(which="both")
    plt.axvspan(start, end, facecolor='red', alpha=.2)
    plt.legend(["sub dataset"], loc="upper left")

for original_entry in sub_dataset:
    for test_input, test_label in test_pairs:
        start_input, end_input = get_start_end(test_input)
        start_label, end_label = get_start_end(test_label)
        plt.figure(figsize=(20,3))
        to_pandas(original_entry).plot(color='royalblue')
        plt.grid(which="both")
        plt.axvspan(start_input, end_input, facecolor='green', alpha=.2)
        plt.axvspan(start_label, end_label, facecolor='blue', alpha=.2)
        plt.legend(["sub dataset"], loc="upper left")
```

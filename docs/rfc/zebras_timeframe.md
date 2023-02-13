# Zebras TimeFrame

We propose ``zebras.TimeFrame``  and ``zebras.Splitframe`` as data structures
to represent and manipulate time series data.

## Status Quo and Problem Statement

GluonTS uses dictionaries to store time series data. Each field can store a
time series or some other data. While this approach is very flexible, one needs
additional knowledge to operate on that data.

For example, this dictionary stores two time series ``target`` and
``feat_dynamic_real`` in addition to some static (time independent data) in
``feat_static_cat``. The two time series have different length, and there is an
implicit assumption that ``target`` is only available in the "past", while
``feat_dynamic_real`` is also available in the "future".

```json
{
    "target": [1, 2, 3, 4],
    "feat_dynamic_real": [1, 2, 3, 4, 5, 6],
    "feat_static_cat": [1, 2],
}
```

Thus, to work with this data one has to know the properties of each field in
addition to some additional context. To exemplify this, we can take a look at
``AddAgeFeature`` which adds a new column to a data-entry, indicating the "age"
of each time step with respect to the beginning of the time series.

```py
def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
    length = target_transformation_length(
        data[self.target_field], self.pred_length, is_train=is_train
    )
    data[self.feature_name] = np.arange(length, dtype=self.dtype)

    return data
```

Here ``target_transformation_length`` returns the length of the time series
by looking at the length of the ``target`` field and adding ``pred_length`` if
``is_train`` is ``True``. Ideally, we should just be able to do something like:

```py
def map_transform(self, data: DataEntry) -> DataEntry:
    length = len(data)
    data[self.feature_name] = np.arange(length, dtype=self.dtype)

    return data
```

Here, we don't need to know about ``target``, ``pred_length`` or ``is_train``
since we can just ask ``data`` about its length directly.


## Proposal

We introduce two new data structures called ``TimeFrame`` and ``SplitFrame`` to
replace our current dictionary based approach.

A ``TimeFrame`` consists of a time index and a set of time series, which all
have the same length. A ``SplitFrame`` consists of two ``TimeFrame`` instances,
one storing data that is ``past`` and one for data that represents ``future``.

A ``SplitFrame`` generally behaves similar to ``TimeFrame`` and the length of
a ``SplitFrame`` is the sum of its ``past`` and ``future`` ``TimeFrame``s.

We can implement many operations directly on ``TimeFrame`` and ``SplitFrame``
to manipulate the underlying data.

For example, ``.set(...)``, ``.remove(...)`` and ``.stack(...)`` add or remove
columns on frame objects. Slicing, as well as ``.pad`` and ``.cat`` alter data
on the time axis. ``Timeframe.split(...)`` can be used to create a
``SplitFrame`` out of a given ``TimeFrame``.

Using these frames, many existing classes in GluonTS such as
``InstanceSplitter`` become almost trivial, since all relevant information and
behaviour is contained in frame objects.

## Discussion

## Comparison with ``pandas.DataFrame``

``TimeFrame`` mimics ``pandas.DataFrame``, however there are some important
differences:

* A ``DataFrame`` can only store one dimensional columns. However, we also need
  support for columns with multiple dimensions, of which one is the time
  dimension.
* In a ``TimeFrame`` all values are evenly spaced with respect to time. This
  then allows us to just extend a ``TimeFrame`` into the future and adjust the
  index automatically.
* We can implement custom behaviour such as ``.split`` easily on ``TimeFrame``.
* ``SplitFrame`` has a notion of past and future.

## Obstacles

Currently, GluonTS is built around unstructured dictionaries for data.
Introducing the proposed classes will require that all components need to be
adapted to ensure that components continue to work together.

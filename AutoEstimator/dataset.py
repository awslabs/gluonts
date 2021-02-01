from gluonts.dataset.artificial import ComplexSeasonalTimeSeries

dataset = ComplexSeasonalTimeSeries(
    num_series=10,
    prediction_length=21,
    freq_str="H",
    length_low=30,
    length_high=200,
    min_val=-10000,
    max_val=10000,
    is_integer=False,
    proportion_missing_values=0,
    is_noise=True,
    is_scale=True,
    percentage_unique_timestamps=1,
    is_out_of_bounds_date=True,
)
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
dataset_m4 = get_dataset("m4_hourly", regenerate=False)

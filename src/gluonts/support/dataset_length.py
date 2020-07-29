from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
print(f"Available datasets: {list(dataset_recipes.keys())}")

# Available datasets:
# ['constant', 'exchange_rate', 'solar-energy', 'electricity', 'traffic', 'exchange_rate_nips',
# 'electricity_nips', 'traffic_nips', 'solar_nips', 'wiki-rolling_nips', 'taxi_30min',
# 'm4_hourly', 'm4_daily', 'm4_weekly', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5']

datasets = ['m4_daily', 'm4_hourly', 'm4_weekly', 'solar-energy', 'electricity']
with open('dataset_length.txt', 'w'):
    pass

with open('dataset_length.txt', 'a') as f:
    for dataset in datasets:
        ds = get_dataset(dataset)
        print(dataset, len(ds.train), len(ds.test))
        f.writelines('\n' + dataset + ' train: ' + str(len(ds.train)) + ' test: ' + str(len(ds.test)))

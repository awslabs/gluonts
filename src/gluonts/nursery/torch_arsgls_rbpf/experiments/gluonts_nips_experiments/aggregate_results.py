import os
from copy import deepcopy
import numpy as np
from box import Box

root_log_path = "/Users/kurler/Desktop/experiment_results"
experiment_log_path = "wiki-rolling_nips/T_4times31"

p = os.path.join(root_log_path, experiment_log_path)
folder_names = tuple(
    filter(lambda f: not os.path.isfile(os.path.join(p, f)), os.listdir(p)))

folder_paths = tuple(os.path.join(p, f) for f in folder_names)

result_names = ["agg_metrics_full_final", "agg_metrics_rolling_final",
                "agg_metrics_full_best", "agg_metrics_rolling_best"]

aggregated_results = {result_name: {} for result_name in result_names}
for result_name in result_names:
    for idx_f, f in enumerate(folder_paths):
        file = np.load(os.path.join(f, "metrics", f"{result_name}.npz"),
                       allow_pickle=True)
        results = file.f.arr_0.item()
        if len(aggregated_results[result_name]) == 0:
            for metric_name in results.keys():
                aggregated_results[result_name][metric_name] = []
        for metric_name, metric in results.items():
            aggregated_results[result_name][metric_name].append(metric)

aggregated_results = Box(aggregated_results)

aggregated_results_mean = Box(deepcopy(aggregated_results))
for result_name in result_names:
    for metric_name in aggregated_results_mean[result_name]:
        aggregated_results_mean[result_name][metric_name] = \
            np.mean(aggregated_results_mean[result_name][metric_name])

aggregated_results_std = Box(deepcopy(aggregated_results))
for result_name in result_names:
    for metric_name in aggregated_results_std[result_name]:
        aggregated_results_std[result_name][metric_name] = \
            np.std(aggregated_results_std[result_name][metric_name])

for metric_name in aggregated_results_std[result_name]:
    for result_name in result_names:
        print(f"{metric_name}: {result_name.split('agg_metrics_')[1]}: "
              f"{aggregated_results_mean[result_name][metric_name]:.4f} "
              f"\pm {aggregated_results_std[result_name][metric_name]:.4f}")

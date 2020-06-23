import os
import json
from gluonts.dataset.field_names import FieldName
import numpy as np

import consts
from data.synthetic_issm import test_utils

# 5 groups where each group has innovation spike on a different day.
gammas = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1, 50, 10],  # Saturday
    [0.1, 0.1, 10.0, 0.1, 0.1, 0.1, 0.1],  # Wednesday
    [0.1, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1],  # Tuesday
    [10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Monday
    [0.1, 0.1, 0.1, 10.0, 0.1, 0.1, 0.1],  # Thursday
]) / 100.0

m0_seasonality = np.array([10.0, 20.0, 25.0, 20.0, 10.0, 100.0, 20.0]) / 100.0
# do not down-scale sigma0. we assume single prior, so make it more diffuse.
sigma0_seasonality = np.array([0.1] * 7)
V0_seasonality = np.diag(sigma0_seasonality ** 2)
# sigmas = [0.1] * 7  # , 0.4, 0.6, 0.8, 1.0, 1.0, 0.7]
sigmas = np.array(
    [sigma * 10 for sigma in [0.1, 0.4, 0.6, 0.8, 1.0, 1.0, 0.7]]) / 100.0
T = 4 * 7
n_steps_forecast = 2 * 7


def generate_seasonality_issm(
        numTsPerGroup,
        T,
        num_groups=5,
        random_prior=True,
        single_source_of_error=False,
        file_path='data.json',
):
    def sigma_t(t):
        return sigmas[t % 7]

    targets = []
    start = '2007-01-01 00:00:00'  # That's a Monday :)
    ts_dict = {'start': start, 'granularity': '1Days'}

    with open(file_path, 'w') as file:
        # num_groups = len(gammas)
        for g in range(num_groups):
            # gamma_t depends on whether it is a weekday or weekend
            gamma = gammas[g]
            a_t_seasonality, F_t_seasonality, g_t_seasonality = \
                test_utils.get_seasonality_issm_coeff(gamma)
            if random_prior:
                l0_seasonality = np.random.multivariate_normal(
                    m0_seasonality, V0_seasonality,
                )
            else:
                l0_seasonality = m0_seasonality

            for i in range(numTsPerGroup):
                target, _ = test_utils.sample_issm(
                    l0_seasonality,
                    a_t_seasonality,
                    g_t_seasonality,
                    sigma_t,
                    T,
                    single_source_of_error,
                )
                targets.append(target)
                ts_dict[FieldName.TARGET] = list(np.squeeze(target))
                ts_dict[FieldName.ITEM_ID] = i
                ts_dict[FieldName.FEAT_STATIC_CAT] = list(np.zeros(1) + g)
                ts_dict[FieldName.FEAT_DYNAMIC_REAL] = None
                json.dump(ts_dict, file)
                file.write("\n")


if __name__ == '__main__':
    for n_data in [20 * i for i in range(1, 11)]:
        print(f"generating dataset with {n_data} samples")
        folder_path = os.path.join(
            consts.data_dir, "synthetic_issm", str(n_data))
        train_folder_path = os.path.join(folder_path, "train")
        test_folder_path = os.path.join(folder_path, "test")

        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(test_folder_path, exist_ok=True)
        generate_seasonality_issm(
            numTsPerGroup=n_data,
            T=T,
            file_path=os.path.join(train_folder_path, "train.json"),
        )
        generate_seasonality_issm(
            numTsPerGroup=n_data,
            T=T + n_steps_forecast,
            file_path=os.path.join(test_folder_path, "test.json"),
        )

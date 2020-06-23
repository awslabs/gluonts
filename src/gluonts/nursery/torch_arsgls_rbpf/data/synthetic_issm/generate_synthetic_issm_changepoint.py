import os
from src import consts
import json
import numpy as np
from gluonts.dataset.field_names import FieldName

from data.synthetic_issm import test_utils

# 5 groups where each group has innovation spike on a different day.
# 1 change-point
gs = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1, 50, 10],  # Saturday
    [0.1, 0.1, 10.0, 0.1, 0.1, 0.1, 0.1],  # Wednesday
    [0.1, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1],  # Tuesday
    [10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Monday
    [0.1, 0.1, 0.1, 10.0, 0.1, 0.1, 0.1],  # Thursday
]) / 100.0
gammas_list = [gs * 1.0, gs * 5.0]

m0_seasonality = np.array([10.0, 20.0, 25.0, 20.0, 10.0, 100.0, 20.0]) / 100.0
# do not down-scale sigma0. we assume single prior, so make it more diffuse.
sigma0_seasonality = np.array([0.1] * 7)
V0_seasonality = np.diag(sigma0_seasonality ** 2)
# sigmas = [0.1] * 7  # , 0.4, 0.6, 0.8, 1.0, 1.0, 0.7]
sigmas = np.array(
    [sigma * 10 for sigma in [0.1, 0.4, 0.6, 0.8, 1.0, 1.0, 0.7]]) / 100.0
change_timesteps = (4 * 7,)
T = 8 * 7
n_steps_forecast = 2 * 7


def generate_seasonality_issm(
        numTsPerGroup,
        T,
        num_groups=5,
        change_timesteps=change_timesteps,
        random_prior=True,
        single_source_of_error=False,
        file_path='data.json',
):
    def sigma_t(t):
        return sigmas[t % 7]

    targets = []
    start = '2007-01-01 00:00:00'  # That's a Monday :)
    ts_dict = {'start': start, 'granularity': '1Days'}

    pattern_durations = np.array(tuple(change_timesteps) + (T,)) \
                        - np.array((0,) + tuple(change_timesteps))

    n_cps = len(gammas_list)
    with open(file_path, 'w') as file:
        # num_groups = len(gammas)
        for g in range(num_groups):
            for i in range(numTsPerGroup):
                target = []
                l0 = np.random.multivariate_normal(
                    m0_seasonality, V0_seasonality) \
                    if random_prior else m0_seasonality
                for idx_cp in range(n_cps):
                    gamma = gammas_list[idx_cp][g]
                    a_t_seasonality, F_t_seasonality, g_t_seasonality = \
                        test_utils.get_seasonality_issm_coeff(gamma)
                    tar, l0 = test_utils.sample_issm(
                        l0=l0,
                        a_t=a_t_seasonality,
                        g_t=g_t_seasonality,
                        sigma_t=sigma_t,
                        t_initial=sum(pattern_durations[:idx_cp]),
                        T=pattern_durations[idx_cp],
                        single_source_of_error=single_source_of_error,
                    )
                    target.append(tar)
                target = np.concatenate(target, axis=0)
                assert target.shape[0] == T
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
            consts.data_dir, "synthetic_issm_changepoint", str(n_data))
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

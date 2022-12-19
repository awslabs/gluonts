"""
The dataset preprocessing is based on DeepAR
https://arxiv.org/pdf/1704.04110.pdf.
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, date
from scipy import stats
import time
import math
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta


class Ele(Dataset):
    def __init__(self, path, train_ins_num, pred_days, overlap, win_len):

        np.random.seed(0)
        train_start = "20110101"
        train_end = "20140831"
        self.covariate_num = 4
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        (
            self.points,
            self.covariates,
            self.dates,
            self.withhold_len,
        ) = self.LoadData(
            path,
            train_start,
            train_end,
            pred_days,
            covariate_num=self.covariate_num,
        )
        print("withhold_len: ", self.withhold_len)
        series_len = self.points.shape[
            0
        ]  # Length of every client's electricity consumption
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        d_hour = 24
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:, i]):
                if j >= 0:
                    break
            indices = range(head + 1, series_len + 1 - (T + self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i, index)
                v = (
                    np.sum(self.points[index : (index + self.win_len), i])
                    / self.win_len
                    + 1
                )
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if count < train_ins_num:
            replace = True
        prob = np.array(self.weight) / sum(self.weight)
        self.dic_keys = np.random.choice(
            range(count), train_ins_num, replace=replace, p=prob
        )

        print(
            "Maxmum traning instances",
            count,
            "Total train instances are: ",
            train_ins_num,
            " Overlap: ",
            overlap,
            " Replace: ",
            replace,
        )

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if type(idx) != int:
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len, points_size))
        train_seq[:, 0] = self.points[
            (series_index - 1) : (series_index + self.win_len - 1), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (self.win_len + series_index), :
        ]
        gt = self.points[series_index : (series_index + T), series_id]
        scaling_factor = self.weight[key]
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)

    def CalCovariate(self, input_time):
        """
        Calculate covariate given an input time string.
        """

        month = input_time.month
        weekday = input_time.weekday()
        hour = input_time.hour
        return np.array([month, weekday, hour])

    def LoadData(self, path, train_start, train_end, pred_days, covariate_num):

        data = pd.read_csv(
            path, sep=";", index_col=0, parse_dates=True, decimal=","
        )
        data = data.resample("1H", label="left", closed="right").sum()

        test_end = datetime.strptime(
            train_end + " 23:00:00", "%Y%m%d %H:%M:%S"
        ) + dt.timedelta(days=pred_days)
        train_start = datetime.strptime(
            train_start + " 00:00:00", "%Y%m%d %H:%M:%S"
        )
        points = data[train_start:test_end].values

        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len = pred_days * d_hour
        withhold_len = pred_len

        dates = data[train_start:test_end].index

        covariates = np.zeros((seq_num, data_len, covariate_num))

        for idx, date in enumerate(dates):
            covariates[:, idx, : (covariate_num - 1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(
                points[:, i]
            ):  # For each time series. we get its first non-zero value' index.
                if j >= 0:
                    break
            if (
                head % d_hour != 0
            ):  # Get first 0:00's index near first non-zero value' index.
                head = head + d_hour - head % d_hour

            covariates[i, head:, covariate_num - 1] = range(
                data_len - head
            )  #  Get its age feature

            for index in range(covariate_num):
                covariates[i, head:, index] = stats.zscore(
                    covariates[i, head:, index]
                )  # We standardize all covariates to have zero mean and unit variance.
        return (points, covariates, dates, withhold_len)


class EleTest(Dataset):
    def __init__(self, points, covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 4
        self.points, self.covariates = (points, covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len // dec_len
        self.test_ins_num = seq_num * rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times, 0, -1):
                index = series_len - enc_len - j * dec_len
                self.sample_index[count] = (
                    i,
                    index,
                )  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]

        train_seq = np.zeros((T, points_size))
        train_seq[: (self.enc_len + 1), 0] = self.points[
            series_index - 1 : (series_index + self.enc_len), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (series_index + T), :
        ]
        gt = self.points[
            series_index + self.enc_len : (series_index + T), series_id
        ]

        scaling_factor = (
            np.sum(
                self.points[
                    series_index : (series_index + self.enc_len), series_id
                ]
            )
            / self.enc_len
            + 1
        )
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)


class Traffic(Dataset):
    def __init__(self, path, train_ins_num, pred_days, overlap, win_len):

        np.random.seed(0)
        self.covariate_num = 4
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        (
            self.points,
            self.covariates,
            self.dates,
            self.withhold_len,
        ) = self.LoadData(path, pred_days, covariate_num=4)
        print("withhold_len: ", self.withhold_len)
        series_len = self.points.shape[
            0
        ]  # Length of every client's electricity consumption
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        d_hour = 24
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:, i]):
                if j > 0:
                    break
            indices = range(head + 1, series_len + 1 - (T + self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i, index)
                v = (
                    np.sum(self.points[index : (index + self.win_len), i])
                    / self.win_len
                    + 1
                )
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if count < train_ins_num:
            replace = True
        prob = np.array(self.weight) / sum(self.weight)
        self.dic_keys = np.random.choice(
            range(count), train_ins_num, replace=replace, p=prob
        )

        print(
            "Maxmum traning instances",
            count,
            "Total train instances are: ",
            train_ins_num,
            " Overlap: ",
            overlap,
            " Replace: ",
            replace,
        )

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if type(idx) != int:
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len, points_size))
        try:
            train_seq[:, 0] = self.points[
                (series_index - 1) : (series_index + self.win_len - 1),
                series_id,
            ]
        except (BaseException):
            import pdb

            pdb.set_trace()
        train_seq[:, 0] = self.points[
            (series_index - 1) : (series_index + self.win_len - 1), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (self.win_len + series_index), :
        ]
        gt = self.points[series_index : (series_index + T), series_id]
        scaling_factor = self.weight[key]
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)

    def CalCovariate(self, input_time):
        month = input_time.month
        weekday = input_time.weekday()
        hour = input_time.hour
        return np.array([month, weekday, hour])

    def LoadData(self, path, pred_days, covariate_num=4):

        data = pd.read_csv(
            path, sep=",", index_col=0, parse_dates=True, decimal="."
        )
        points = data.values
        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len = pred_days * d_hour
        withhold_len = pred_len

        dates = data.index

        covariates = np.zeros((seq_num, data_len, covariate_num))

        for idx, date in enumerate(dates):
            covariates[:, idx, : (covariate_num - 1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(
                points[:, i]
            ):  # For each time series. we get its first non-zero value' index.
                if j > 0:
                    break
            covariates[i, head:, covariate_num - 1] = range(
                data_len - head
            )  #  Get its age feature

            for index in range(covariate_num):
                covariates[i, head:, index] = stats.zscore(
                    covariates[i, head:, index]
                )  # We standardize all covariates to have zero mean and unit variance.
        return (points, covariates, dates, withhold_len)


class TrafficTest(Dataset):
    def __init__(self, points, covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 4
        self.points, self.covariates = (points, covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len // dec_len
        self.test_ins_num = seq_num * rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times, 0, -1):
                index = series_len - enc_len - j * dec_len
                self.sample_index[count] = (
                    i,
                    index,
                )  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T, points_size))

        train_seq[: (self.enc_len + 1), 0] = self.points[
            series_index - 1 : (series_index + self.enc_len), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (series_index + T), :
        ]

        gt = self.points[
            series_index + self.enc_len : (series_index + T), series_id
        ]

        scaling_factor = (
            np.sum(
                self.points[
                    series_index : (series_index + self.enc_len), series_id
                ]
            )
            / self.enc_len
            + 1
        )
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)


class Traffic_fine(Dataset):
    def __init__(self, path, train_ins_num, pred_days, overlap, win_len):

        np.random.seed(0)
        self.covariate_num = 5
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        (
            self.points,
            self.covariates,
            self.dates,
            self.withhold_len,
        ) = self.LoadData(path, pred_days, covariate_num=5)
        print("withhold_len: ", self.withhold_len)
        series_len = self.points.shape[0]
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:, i]):
                if j > 0:
                    break
            indices = range(head + 1, series_len + 1 - (T + self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i, index)
                v = (
                    np.sum(self.points[index : (index + self.win_len), i])
                    / self.win_len
                    + 1
                )
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if count < train_ins_num:
            replace = True
        prob = np.array(self.weight) / sum(self.weight)
        self.dic_keys = np.random.choice(
            range(count), train_ins_num, replace=replace, p=prob
        )

        print(
            "Maxmum traning instances",
            count,
            "Total train instances are: ",
            train_ins_num,
            " Overlap: ",
            overlap,
            " Replace: ",
            replace,
        )

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if type(idx) != int:
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len, points_size))
        try:
            train_seq[:, 0] = self.points[
                (series_index - 1) : (series_index + self.win_len - 1),
                series_id,
            ]
        except (BaseException):
            import pdb

            pdb.set_trace()
        train_seq[:, 0] = self.points[
            (series_index - 1) : (series_index + self.win_len - 1), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (self.win_len + series_index), :
        ]
        gt = self.points[series_index : (series_index + T), series_id]

        scaling_factor = self.weight[key]
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)

    def CalCovariate(self, input_time):
        month = input_time.month
        weekday = input_time.weekday()
        hour = input_time.hour
        minute = input_time.minute
        return np.array([month, weekday, hour, minute])

    def LoadData(self, path, pred_days, covariate_num):

        data = pd.read_csv(
            path, sep=",", index_col=0, parse_dates=True, decimal="."
        )
        data = data.resample("20T", label="left", closed="left").mean()
        data = data.fillna(0)
        points = data.values
        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len = pred_days * d_hour * 3
        withhold_len = pred_len

        dates = data.index

        covariates = np.zeros((seq_num, data_len, covariate_num))

        for idx, date in enumerate(dates):
            covariates[:, idx, : (covariate_num - 1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(
                points[:, i]
            ):  # For each time series. we get its first non-zero value' index.
                if j > 0:
                    break
            covariates[i, head:, covariate_num - 1] = range(
                data_len - head
            )  #  Get its age feature

            for index in range(covariate_num):
                covariates[i, head:, index] = stats.zscore(
                    covariates[i, head:, index]
                )  # We standardize all covariates to have zero mean and unit variance.
        return (points, covariates, dates, withhold_len)


class Traffic_fineTest(Dataset):
    def __init__(self, points, covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 5
        self.points, self.covariates = (points, covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len // dec_len
        self.test_ins_num = seq_num * rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times, 0, -1):
                index = series_len - enc_len - j * dec_len
                self.sample_index[count] = (
                    i,
                    index,
                )  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T, points_size))
        train_seq[: (self.enc_len + 1), 0] = self.points[
            series_index - 1 : (series_index + self.enc_len), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (series_index + T), :
        ]
        gt = self.points[
            series_index + self.enc_len : (series_index + T), series_id
        ]
        scaling_factor = (
            np.sum(
                self.points[
                    series_index : (series_index + self.enc_len), series_id
                ]
            )
            / self.enc_len
            + 1
        )
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)


class Ele_fine(Dataset):
    def __init__(self, path, train_ins_num, pred_days, overlap, win_len):

        np.random.seed(0)
        train_start = "20110101"
        train_end = "20140831"
        self.covariate_num = 5
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        (
            self.points,
            self.covariates,
            self.dates,
            self.withhold_len,
        ) = self.LoadData(
            path,
            train_start,
            train_end,
            pred_days,
            covariate_num=self.covariate_num,
        )
        print("withhold_len: ", self.withhold_len)
        series_len = self.points.shape[
            0
        ]  # Length of every client's electricity consumption
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:, i]):
                if j >= 0:
                    break
            indices = range(head + 1, series_len + 1 - (T + self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i, index)
                v = (
                    np.sum(self.points[index : (index + self.win_len), i])
                    / self.win_len
                    + 1
                )
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if count < train_ins_num:
            replace = True
        prob = np.array(self.weight) / sum(self.weight)
        self.dic_keys = np.random.choice(
            range(count), train_ins_num, replace=replace, p=prob
        )

        print(
            "Maxmum traning instances",
            count,
            "Total train instances are: ",
            train_ins_num,
            " Overlap: ",
            overlap,
            " Replace: ",
            replace,
        )

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if type(idx) != int:
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len, points_size))
        try:
            train_seq[:, 0] = self.points[
                (series_index - 1) : (series_index + self.win_len - 1),
                series_id,
            ]
        except (BaseException):
            import pdb

            pdb.set_trace()
        train_seq[:, 0] = self.points[
            (series_index - 1) : (series_index + self.win_len - 1), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (self.win_len + series_index), :
        ]
        gt = self.points[series_index : (series_index + T), series_id]

        scaling_factor = self.weight[key]
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)

    def CalCovariate(self, input_time):
        """
        Calculate covariate given an input time string.
        """

        month = input_time.month
        weekday = input_time.weekday()
        hour = input_time.hour
        minute = input_time.minute
        return np.array([month, weekday, hour, minute])

    def LoadData(self, path, train_start, train_end, pred_days, covariate_num):

        data = pd.read_csv(
            path, sep=";", index_col=0, parse_dates=True, decimal=","
        )
        test_end = datetime.strptime(
            train_end + " 00:00:00", "%Y%m%d %H:%M:%S"
        ) + dt.timedelta(days=pred_days)
        train_start = datetime.strptime(
            train_start + " 00:15:00", "%Y%m%d %H:%M:%S"
        )
        points = data[train_start:test_end].values

        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len = pred_days * d_hour * 4
        withhold_len = pred_len

        dates = data[train_start:test_end].index

        covariates = np.zeros((seq_num, data_len, covariate_num))

        for idx, date in enumerate(dates):
            covariates[:, idx, : (covariate_num - 1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(
                points[:, i]
            ):  # For each time series. we get its first non-zero value' index.
                if j >= 0:
                    break

            covariates[i, head:, covariate_num - 1] = range(
                data_len - head
            )  #  Get its age feature

            for index in range(covariate_num):
                covariates[i, head:, index] = stats.zscore(
                    covariates[i, head:, index]
                )  # We standardize all covariates to have zero mean and unit variance.
        return (points, covariates, dates, withhold_len)


class Ele_fineTest(Dataset):
    def __init__(self, points, covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 5
        self.points, self.covariates = (points, covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len // dec_len
        self.test_ins_num = seq_num * rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times, 0, -1):
                index = series_len - enc_len - j * dec_len
                self.sample_index[count] = (
                    i,
                    index,
                )  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T, points_size))
        train_seq[: (self.enc_len + 1), 0] = self.points[
            series_index - 1 : (series_index + self.enc_len), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (series_index + T), :
        ]
        gt = self.points[
            series_index + self.enc_len : (series_index + T), series_id
        ]
        scaling_factor = (
            np.sum(
                self.points[
                    series_index : (series_index + self.enc_len), series_id
                ]
            )
            / self.enc_len
            + 1
        )
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)


class M4(Dataset):
    def __init__(self, path, train_ins_num, pred_days, overlap, win_len):

        np.random.seed(0)
        self.covariate_num = 3
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        (
            self.points,
            self.covariates,
            self.dates,
            self.withhold_len,
        ) = self.LoadData(path, pred_days, covariate_num=self.covariate_num)
        print("withhold_len: ", self.withhold_len)
        series_len = self.points.shape[0]
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        d_hour = 24
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            if self.points[0, i] < 0:
                indices = range(261, series_len + 1 - (T + self.withhold_len))
            else:
                indices = range(1, series_len + 1 - (T + self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i, index)
                v = (
                    np.sum(self.points[index : (index + self.win_len), i])
                    / self.win_len
                    + 1
                )
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if count < train_ins_num:
            replace = True
        prob = np.array(self.weight) / sum(self.weight)
        self.dic_keys = np.random.choice(
            range(count), train_ins_num, replace=replace, p=prob
        )

        print(
            "Maxmum traning instances",
            count,
            "Total train instances are: ",
            train_ins_num,
            " Overlap: ",
            overlap,
            " Replace: ",
            replace,
        )

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if type(idx) != int:
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len, points_size))
        try:
            train_seq[:, 0] = self.points[
                (series_index - 1) : (series_index + self.win_len - 1),
                series_id,
            ]
        except (BaseException):
            import pdb

            pdb.set_trace()

        train_seq[:, 0] = self.points[
            (series_index - 1) : (series_index + self.win_len - 1), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (self.win_len + series_index), :
        ]
        gt = self.points[series_index : (series_index + T), series_id]

        scaling_factor = self.weight[key]
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)

    def CalCovariate(self, input_time):
        """
        Calculate covariate given an input time string.
        """

        weekday = input_time.weekday()
        hour = input_time.hour
        return np.array([weekday, hour])

    def LoadData(self, path, pred_days, covariate_num):
        data = pd.read_csv(
            path, sep=",", index_col=0, parse_dates=True, decimal="."
        )
        points = np.transpose(data.values)
        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len = pred_days * d_hour
        withhold_len = pred_len
        dates = data.index
        covariates = np.zeros((seq_num, data_len, covariate_num))
        head = 0
        for idx, date in enumerate(dates):
            if points[0, idx] < 0:
                head = 260
            else:
                head = 0
            for idx_date in range(data_len - head):
                covariates[
                    idx, idx_date + head, : (covariate_num - 1)
                ] = self.CalCovariate(date + dt.timedelta(hours=idx_date))
            covariates[idx, head:, covariate_num - 1] = range(data_len - head)
            for index in range(covariate_num):
                covariates[idx, head:, index] = stats.zscore(
                    covariates[idx, head:, index]
                )  # We standardize all covariates to have zero mean and unit variance.
        return (points, covariates, dates, withhold_len)


class M4Test(Dataset):
    def __init__(self, points, covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 3
        self.points, self.covariates = (points, covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len // dec_len
        self.test_ins_num = seq_num * rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times, 0, -1):
                index = series_len - enc_len - j * dec_len
                self.sample_index[count] = (
                    i,
                    index,
                )  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T, points_size))
        train_seq[: (self.enc_len + 1), 0] = self.points[
            series_index - 1 : (series_index + self.enc_len), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (series_index + T), :
        ]
        gt = self.points[
            series_index + self.enc_len : (series_index + T), series_id
        ]
        scaling_factor = (
            np.sum(
                self.points[
                    series_index : (series_index + self.enc_len), series_id
                ]
            )
            / self.enc_len
            + 1
        )
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)


class Wind(Dataset):
    def __init__(self, path, train_ins_num, pred_days, overlap, win_len):

        np.random.seed(0)
        self.covariate_num = 4
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        (
            self.points,
            self.covariates,
            self.dates,
            self.withhold_len,
        ) = self.LoadData(path, pred_days, covariate_num=4)
        print("withhold_len: ", self.withhold_len)
        series_len = self.points.shape[0]
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:, i]):
                if j > 0:
                    break
            indices = range(head + 1, series_len + 1 - (T + self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i, index)
                v = (
                    np.sum(self.points[index : (index + self.win_len), i])
                    / self.win_len
                    + 1
                )
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if count < train_ins_num:
            replace = True
        prob = np.array(self.weight) / sum(self.weight)
        self.dic_keys = np.random.choice(
            range(count), train_ins_num, replace=replace, p=prob
        )

        print(
            "Maxmum traning instances",
            count,
            "Total train instances are: ",
            train_ins_num,
            " Overlap: ",
            overlap,
            " Replace: ",
            replace,
        )

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if type(idx) != int:
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len, points_size))
        try:
            train_seq[:, 0] = self.points[
                (series_index - 1) : (series_index + self.win_len - 1),
                series_id,
            ]
        except (BaseException):
            import pdb

            pdb.set_trace()
        train_seq[:, 0] = self.points[
            (series_index - 1) : (series_index + self.win_len - 1), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (self.win_len + series_index), :
        ]
        gt = self.points[series_index : (series_index + T), series_id]
        scaling_factor = self.weight[key]
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)

    def CalCovariate(self, input_time):
        year = input_time.year
        month = input_time.month
        day = input_time.day
        return np.array([year, month, day])

    def LoadData(self, path, pred_days, covariate_num=4):

        data = pd.read_csv(
            path, sep=",", index_col=0, parse_dates=True, decimal="."
        )
        points = data.values
        seq_num = points.shape[1]
        data_len = points.shape[0]
        pred_len = pred_days
        withhold_len = pred_len
        dates = data.index
        covariates = np.zeros((seq_num, data_len, covariate_num))

        for idx, date in enumerate(dates):
            covariates[:, idx, : (covariate_num - 1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(
                points[:, i]
            ):  # For each time series. we get its first non-zero value' index.
                if j >= 0:
                    break
            covariates[i, head:, covariate_num - 1] = range(
                data_len - head
            )  #  Get its age feature

            for index in range(covariate_num):
                result = stats.zscore(
                    covariates[i, head:, index]
                )  # We standardize all covariates to have zero mean and unit variance.
                if np.isnan(result).any():
                    print(points[:, i])
                    import pdb

                    pdb.set_trace()

                else:
                    covariates[i, head:, index] = result
        return (points, covariates, dates, withhold_len)


class WindTest(Dataset):
    def __init__(self, points, covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 4
        self.points, self.covariates = (points, covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len // dec_len
        self.test_ins_num = seq_num * rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times, 0, -1):
                index = series_len - enc_len - j * dec_len
                self.sample_index[count] = (
                    i,
                    index,
                )  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T, points_size))
        train_seq[: (self.enc_len + 1), 0] = self.points[
            series_index - 1 : (series_index + self.enc_len), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (series_index + T), :
        ]

        gt = self.points[
            series_index + self.enc_len : (series_index + T), series_id
        ]
        scaling_factor = (
            np.sum(
                self.points[
                    series_index : (series_index + self.enc_len), series_id
                ]
            )
            / self.enc_len
            + 1
        )
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor

        return (train_seq, gt, series_id, scaling_factor)


class Solar(Dataset):
    def __init__(self, path, train_ins_num, pred_days, overlap, win_len):

        np.random.seed(0)
        self.covariate_num = 4
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        (
            self.points,
            self.covariates,
            self.dates,
            self.withhold_len,
        ) = self.LoadData(path, pred_days, covariate_num=4)
        print("withhold_len: ", self.withhold_len)
        series_len = self.points.shape[0]
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        d_hour = 24
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:, i]):
                if j > 0:
                    break
            indices = range(head + 1, series_len + 1 - (T + self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i, index)
                v = (
                    np.sum(self.points[index : (index + self.win_len), i])
                    / self.win_len
                    + 1
                )
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if count < train_ins_num:
            replace = True
        prob = np.array(self.weight) / sum(self.weight)
        self.dic_keys = np.random.choice(
            range(count), train_ins_num, replace=replace, p=prob
        )

        print(
            "Maxmum traning instances",
            count,
            "Total train instances are: ",
            train_ins_num,
            " Overlap: ",
            overlap,
            " Replace: ",
            replace,
        )

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if type(idx) != int:
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len, points_size))
        try:
            train_seq[:, 0] = self.points[
                (series_index - 1) : (series_index + self.win_len - 1),
                series_id,
            ]
        except (BaseException):
            import pdb

            pdb.set_trace()
        train_seq[:, 0] = self.points[
            (series_index - 1) : (series_index + self.win_len - 1), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (self.win_len + series_index), :
        ]
        gt = self.points[series_index : (series_index + T), series_id]

        scaling_factor = self.weight[key]
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor
        return (train_seq, gt, series_id, scaling_factor)

    def CalCovariate(self, input_time):
        month = input_time.month
        weekday = input_time.weekday()
        hour = input_time.hour
        return np.array([month, weekday, hour])

    def LoadData(self, path, pred_days, covariate_num=4):

        data = pd.read_csv(
            path, sep=",", index_col=0, parse_dates=True, decimal="."
        )
        points = data.values
        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len = pred_days * d_hour
        withhold_len = pred_len

        dates = data.index

        covariates = np.zeros((seq_num, data_len, covariate_num))

        for idx, date in enumerate(dates):
            covariates[:, idx, : (covariate_num - 1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(
                points[:, i]
            ):  # For each time series. we get its first non-zero value' index.
                if j > 0:
                    break
            covariates[i, head:, covariate_num - 1] = range(
                data_len - head
            )  #  Get its age feature

            for index in range(covariate_num):
                covariates[i, head:, index] = stats.zscore(
                    covariates[i, head:, index]
                )  # We standardize all covariates to have zero mean and unit variance.
        return (points, covariates, dates, withhold_len)


class SolarTest(Dataset):
    def __init__(self, points, covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 4
        self.points, self.covariates = (points, covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len // dec_len
        self.test_ins_num = seq_num * rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times, 0, -1):
                index = series_len - enc_len - j * dec_len
                self.sample_index[count] = (
                    i,
                    index,
                )  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T, points_size))
        train_seq[: (self.enc_len + 1), 0] = self.points[
            series_index - 1 : (series_index + self.enc_len), series_id
        ]
        train_seq[:, 1:] = self.covariates[
            series_id, series_index : (series_index + T), :
        ]
        gt = self.points[
            series_index + self.enc_len : (series_index + T), series_id
        ]
        scaling_factor = (
            np.sum(
                self.points[
                    series_index : (series_index + self.enc_len), series_id
                ]
            )
            / self.enc_len
            + 1
        )
        train_seq[:, 0] = train_seq[:, 0] / scaling_factor

        return (train_seq, gt, series_id, scaling_factor)

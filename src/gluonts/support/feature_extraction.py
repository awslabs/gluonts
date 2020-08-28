from autogluon import TabularPrediction as task
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd


class Gluonts2Auto:
    def __init__(self, dataset_name):
        self.dataset = get_dataset(dataset_name, regenerate=False)
        self.train_data, self.val_data, self.test_data = self.conversion()

    def conversion(self):
        train_series, test_series = to_pandas(next(iter(self.dataset.train))), to_pandas(
            next(iter(self.dataset.test)))
        total = pd.concat([train_series, test_series])
        a = len(total) // 10
        b, c = a * 8, a * 9
        train_series, val_series, test_series = total[:b], total[b:c], total[c:]
        train_df, val_df, test_df = self.parse_series(train_series), self.parse_series(val_series), self.parse_series(
            test_series)
        return task.Dataset(train_df), task.Dataset(val_df), task.Dataset(test_df)

    @staticmethod
    def parse_series(series):
        hour_of_day = series.index.hour
        month_of_year = series.index.month
        day_of_week = series.index.dayofweek
        year_idx = series.index.year
        target = series.values
        cal = calendar()
        holidays = cal.holidays(start=series.index.min(), end=series.index.max())
        df = pd.DataFrame(zip(year_idx, month_of_year, day_of_week, hour_of_day, series.index.isin(holidays), target),
                          columns=['year_idx', 'month_of_year', 'day_of_week', 'hour_of_day', 'holiday', 'target'])
        convert_type = {x: 'category' for x in df.columns.values[:-2]}
        df = df.astype(convert_type)
        return df


if __name__ == '__main__':
    g = Gluonts2Auto('electricity')
    g.conversion()

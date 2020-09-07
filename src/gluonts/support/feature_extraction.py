from autogluon import TabularPrediction as task
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd


class Gluonts2Auto:

    '''
    Input:
    dataset_name: str, the name of the gluonts available dataset
    regenerate: boolean, the parameter at get_dataset method
    sample call:
    g = Gluonts2Auto('electricity')
    train_task, test_task = g.adapter()
    '''

    def __init__(self, dataset_name, regenerate=True):
        self.dataset = get_dataset(dataset_name, regenerate=regenerate)

    def adapter(self):
        train_ts, test_ts = self.dataset.train, self.dataset.test
        train_series, test_series = self.get_series(train_ts), self.get_series(test_ts)
        train_df, test_df = self.parse_series(train_series), self.parse_series(test_series)
        return task.Dataset(train_df), task.Dataset(test_df)

    @staticmethod
    def get_series(dataset):
        series = pd.Series(dtype='float32')
        for i in range(len(dataset)):
            entry = to_pandas(list(dataset)[i])
            series = series.append(entry)
        return series

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


g = Gluonts2Auto('electricity')
train_task, test_task = g.adapter()
print(train_task)
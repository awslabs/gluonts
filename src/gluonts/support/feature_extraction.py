from autogluon import TabularPrediction as task
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd

'''
class Gluonts2Auto:
    def __init__(self, dataset_name):
        self.dataset = get_dataset(dataset_name, regenerate=False)
        self.train_data, self.val_data, self.test_data = self.conversion()

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
    '''

'''
class Gluonts2Auto:
    def __init__(self, dataset_name, problem_type):
        self.dataset = get_dataset(dataset_name, regenerate=False)
        self.train_data, self.val_data, self.test_data = self.conversion()
        self.problem_type = problem_type
        self.freq = self.dataset.metadata.freq

    def conversion(self):
        train_series, test_series = to_pandas(list(iter(self.dataset.train))), to_pandas(
            list(iter(self.dataset.test)))
        print(train_series)
        total = pd.concat([train_series, test_series])
        a = len(total) // 10
        b, c = a*8, a*9
        train_series, val_series, test_series = total[:b], total[b:c], total[c:]
        train_df, val_df, test_df = self.parse_series(train_series),self.parse_series(val_series), self.parse_series(test_series)
        return task.Dataset(train_df), task.Dataset(val_df), task.Dataset(test_df)

    def get_pandas_ls(self, dataset):
        ls = []
        for ts in list(dataset.test):
            ls.append(to_pandas(ts))
        return ls

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

    def test(self):
        print(self.dataset.metadata)
        print(len(to_pandas(next(iter(self.dataset.test))))-len(to_pandas(next(iter(self.dataset.train)))))

    def change_test(self):
        for data_entry in self.convert_df(iter(self.dataset.test)):
            yield data_entry["ts"]

    def convert_df(self, data_iterator):
        for data_entry in data_iterator:
            data = data_entry.copy()
            index = pd.date_range(
                start=data["start"],
                freq=self.freq,
                periods=data["target"].shape[-1],
            )
            data["ts"] = pd.DataFrame(
                index=index, data=data["target"].transpose()
            )
            yield data

    def change_forecast(self, y_pred):
        forecasts = []
        entry = next(iter(self.dataset.test))
        test_series = to_pandas(entry)
        time_index = test_series.index
        ts = pd.Series(y_pred, index=time_index)
        i = 0
        for i in range(len(self.dataset.test)):
            forecast = SampleForecast(freq=self.freq, )



    def evaluate_it(self):
        label_column = 'target'
        predictor = task.fit(train_data=self.train_data, tuning_data=self.val_data, label=label_column,
                             problem_type=self.problem_type,
                             output_directory='AutogluonModels/ag-a')
        # print(predictor.transform_features())
        y_test = self.test_data[label_column]
        test_data_nolab = self.test_data.drop(labels=[label_column], axis=1)
        y_pred = predictor.predict(test_data_nolab)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        # conver the test dataset
        ts_it = self.change_test()
        tss = list(ts_it)
        # convert the forecasts
        print(len(y_pred))
        print(len(tss))
        # forecast_it = self.change_forecast(y_pred)
        forecasts = list(forecast_it)
        # agg_metrics, item_metrics = evaluator(iter(tss), iter(y_pred), num_series=len(self.test_data))
        # return json.dumps(agg_metrics, indent=4), self.y_pred, self.y_test
        # return y_pred

    def plot_result(self):
        plt.figure(figsize=(17, 9))
        plt.grid(True)
        plt.plot(self.y_pred[:100], color="red", label='y_pred')
        plt.plot(self.y_test[:100], color="green", label='y')
        plt.legend()
        plt.title('Prediction Results')
        plt.show()



if __name__ == '__main__':
    g = Gluonts2Auto('electricity', 'regression')
    # print(list(g.change_test()))
    g.conversion()
    '''



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
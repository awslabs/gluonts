from autogluon import TabularPrediction as task
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd
import matplotlib.pyplot as plt


class Gluonts2Auto:
    def __init__(self, dataset_name):
        self.dataset = self.get_ds(dataset_name)

    @staticmethod
    def get_ds(dataset_name):
        return get_dataset(dataset_name, regenerate=False)

    def conversion(self):
        train_series, test_series = to_pandas(next(iter(self.dataset.train))), to_pandas(
            next(iter(self.dataset.test)))
        train_series = self.sample_train(train_series)
        train_df, test_df = self.parse_series(train_series), self.parse_series(test_series)
        return task.Dataset(train_df), task.Dataset(test_df)
        # train_series1, train_series2 = self.split_train(train_series)
        # train_df1, train_df2, test_df = self.parse_series(train_series1), self.parse_series(train_series2), self.parse_series(test_series)
        # return task.Dataset(train_df1), task.Dataset(train_df2), task.Dataset(test_df)

    def sample_train(self, series):
        l = - (-series.size // 5)
        new_series = pd.core.series.Series(dtype='float64')
        for i in range(l-1):
            subseries = series[i*5:(i+1)*5].sample(1)
            new_series = new_series.append(subseries)
        return new_series

# get a random data point from each 5 points to reduce overfitting.
    def split_train(self, series):
        mini, maxi = series.values.min(), series.values.max()
        splitter = (mini+maxi) // 2
        train1, train2 = series.loc[lambda series: series < splitter], series.loc[lambda series: series >= splitter]
        print(train1.head(10), train2.head(10))
        # train1.plot()
        # plt.grid(which="both")
        # plt.legend(["train series"], loc="upper left")
        # plt.title("The smaller set")
        # plt.show()
        #
        # train2.plot()
        # plt.grid(which="both")
        # plt.legend(["train series"], loc="upper left")
        # plt.title("The larger set")
        # plt.show()

        return train1, train2

    @staticmethod
    def parse_series(series):
        hour_of_day = series.index.hour
        month_of_year = series.index.month
        day_of_week = series.index.dayofweek
        year_idx = series.index.year
        week_of_year = series.index.week
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
    # train_data1, train_data2, test_data = g.conversion()
    train_data, test_data = g.conversion()
    label_column = 'target'
    predictor = task.fit(train_data=train_data, label=label_column, problem_type='regression', output_directory='AutogluonModels/ag-a')
    # predictor = task.fit(train_data=train_data1, label=label_column, problem_type='regression', output_directory='AutogluonModels/ag-a')
    # predictor1 = task.fit(train_data=train_data2, label=label_column, problem_type='regression', output_directory='AutogluonModels/ag-b')
    y_test = test_data[label_column]
    test_data_nolab = test_data.drop(labels=[label_column], axis=1)
    y_pred = predictor.predict(test_data_nolab)

    print("Predictions:", y_pred)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(perf)

    plt.figure(figsize=(17, 9))
    plt.grid(True)
    plt.plot(y_pred[:100], color="red", label='y_pred')
    plt.plot(y_test[:100], color="green", label='y')
    plt.legend()
    plt.title('Prediction Results')
    plt.show()




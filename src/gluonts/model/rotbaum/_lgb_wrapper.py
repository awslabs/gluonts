import pandas as pd

from lightgbm import LGBMRegressor


class lgb_wrapper:
    """
    A wrapped of lightgbm that can be fed into the model parameters in QRX
    and TreePredictor.
    """
    def __init__(self, **lgb_params):
        self.model = LGBMRegressor(**lgb_params)

    def fit(self, train_data, train_target):
        self.model.fit(pd.DataFrame(train_data), train_target)

    def predict(self, data):
        return self.model.predict(pd.DataFrame(data))

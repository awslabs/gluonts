# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost

from gluonts.core.component import validated


class QRF:
    @validated()
    def __init__(self, params: Optional[dict] = None):
        """
        Implements Quantile Random Forests using skgarden.
        """
        from skgarden import RandomForestQuantileRegressor

        self.model = RandomForestQuantileRegressor(**params)

    def fit(self, x_train, y_train):
        self.model.fit(np.array(x_train), np.array(y_train))

    def predict(self, x_test, quantile):
        return self.model.predict(x_test, quantile=100 * quantile)


class QuantileReg:
    @validated()
    def __init__(self, quantiles: List, params: Optional[dict] = None):
        """
        Implements quantile regression using lightgbm.
        """
        from lightgbm import LGBMRegressor

        self.quantiles = quantiles
        self.models = dict(
            (
                quantile,
                LGBMRegressor(objective="quantile", alpha=quantile, **params),
            )
            for quantile in quantiles
        )

    def fit(self, x_train, y_train):
        for model in self.models.values():
            model.fit(np.array(x_train), np.array(y_train))

    def predict(self, x_test, quantile):
        return self.models[quantile].predict(x_test)


class QRX:
    @validated()
    def __init__(
        self,
        model=None,
        xgboost_params: Optional[dict] = None,
        clump_size: int = 100,
    ):
        """
        QRX is an algorithm that takes a point estimate algorithm and turns it
        into a probabilistic forecasting algorithm. By default it uses XGBoost.

        You fit it once, and choose the quantile to predict only at
        prediction time.

        Prediction is done by taking empirical quantiles of *true values*
        associated with point estimate predictions close to the point
        estimate of the given point. The minimal number of associated true
        values is determined by clump_size.

        The algorithm is (loosely) inspired by quantile regression
        forests, in that it is predicts quantiles based on associated true
        values, where the association is based on a point estimate algorithm.

        Parameters
        ----------
        model
            Any point estimate algorithm with .fit and .predict functions.
        xgboost_params
            If None, then it uses
            {"max_depth": 5, "n_jobs": -1, "verbosity": 1,
             "objective": "reg:squarederror"}
        clump_size
            Hyperparameter that determines the minimal size of the list of
            true values associated with each prediction.
        """
        if model:
            self.model = model
        else:
            self.model = self._create_xgboost_model(xgboost_params)
        self.clump_size = clump_size
        self.df = None
        self.processed_df = None
        self.cell_values = None
        self.cell_values_dict = None
        self.quantile_dicts = {}

    @staticmethod
    def _create_xgboost_model(model_params: Optional[dict] = None):
        """
        Creates an xgboost model using specified or default parameters.
        """
        if model_params is None:
            model_params = {
                "max_depth": 5,
                "n_jobs": -1,
                "verbosity": 1,
                "objective": "reg:squarederror",
            }
        return xgboost.sklearn.XGBModel(**model_params)

    def fit(self, x_train, y_train):
        """
        Fits self.model and partitions R^n into cells. More accurately,
        it creates a dictionary taking predictions on train to lists of
        associated true values, and puts it in self.cell_values.

        Parameters
        ----------
        x_train: list
            list of lists
        y_train: list
        """
        self.quantile_dicts = {}
        x_train, y_train = np.array(x_train), np.array(y_train)  # xgboost
        # doens't like lists
        self.model.fit(np.array(x_train), np.array(y_train))
        y_train_pred = self.model.predict(x_train)
        self.df = pd.DataFrame(
            {"x": list(x_train), "y_true": y_train, "y_pred": y_train_pred}
        )
        self.cell_values_dict = self.preprocess_df(
            self.df, clump_size=self.clump_size
        )
        self.cell_values = sorted(self.cell_values_dict.keys())

    @staticmethod
    def clump(dic: Dict, min_num: int) -> Dict:
        """
        Returns a new dictionary whose keys are the same as dic's keys.
        Runs over dic's keys, from smallest to largest, and every time that
        the sum of the lengths of the values goes over min_num, it makes the
        new dictionary's values for the associated keys reference a single
        list object whose elements are the with-multiplicity union of the
        lists that appear as values in dic.

        Note that in the dictionary that is being output by this function,
        while the keys are the same number of keys as in dic, the number of
        objects in the values can be significantly smaller.

        Examples:
        >>> QRX.clump({0.1: [3, 3], 0.3: [0], 1.5: [-8]}, 0)
        {0.1: [3, 3], 0.3: [0], 1.5: [-8]}

        >>> QRX.clump({0.1: [3, 3], 0.3: [0], 1.5: [-8]}, 1)
        {0.1: [3, 3], 0.3: [0, -8], 1.5: [0, -8]}

        >>> QRX.clump({0.1: [3, 3], 0.3: [0], 1.5: [-8]}, 2)
        {0.1: [3, 3, 0], 0.3: [3, 3, 0], 1.5: [-8]}

        Parameters
        ----------
        dic: dict
            float to list
        min_num: int
            minimal number of clump size.

        Returns
        -------
        dict
            float to list; with the values often having the same list object
            appear multiple times
        """
        sorted_keys = sorted(dic)
        new_dic = {}
        iter_length = 0
        iter_list = []
        for key in sorted_keys:
            iter_length += len(dic[key])
            iter_list.extend(dic[key])
            new_dic[key] = iter_list  # Note that iter_list may change in the
            # future, and this will change the value of new_dic[key]. This
            # is intentional.
            if iter_length > min_num:
                iter_length = 0
                iter_list = []  # This line, of course, doesn't change any
                # value of new_dic, as it makes iter_list reference a new
                # list object.
        return new_dic

    @classmethod
    def preprocess_df(cls, df: pd.DataFrame, clump_size: int = 100) -> Dict:
        """
        Associates true values to each prediction that appears in train. For
        the nature of this association, see details in .clump.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with columns 'y_true' and 'y_pred', of true and
            predicted values on the training set.
        clump_size
            Size of clumps to associate to each prediction in the set of
            predictions on the training set.

        Returns
        -------
        dict
            going from predictions from the set of predictions on the
            training set to lists of associated true values, with the length
            of each being at least clump_size.
        """
        dic = dict(df.groupby("y_pred")["y_true"].apply(list))
        dic = cls.clump(dic, clump_size)
        return dic

    @classmethod
    def get_closest_pt(cls, sorted_list: List, num: int) -> int:
        """
        Given a sorted list of floats, returns the number closest to num.
        Implements a binary search.
        """
        assert sorted_list
        if len(sorted_list) == 1:
            return sorted_list[0]
        else:
            halfway_indx = (len(sorted_list) - 1) // 2
            if sorted_list[halfway_indx] > num:
                return cls.get_closest_pt(sorted_list[: halfway_indx + 1], num)
            elif sorted_list[halfway_indx + 1] < num:
                return cls.get_closest_pt(sorted_list[halfway_indx + 1 :], num)
            elif abs(sorted_list[halfway_indx] - num) < abs(
                sorted_list[halfway_indx + 1] - num
            ):
                return sorted_list[halfway_indx]
            else:
                return sorted_list[halfway_indx + 1]

    @staticmethod
    def _get_quantiles_from_dic_with_list_values(
        dic: Dict, quantile: float
    ) -> Dict:
        """
        Given a dictionary of float to lists, returns a
        dictionary that takes num to np.percentile(dic[num], 100 * quantile).

        The function is meant to be efficient under the assumption that
        dic's values have list objects that repeat many times over.

        Parameters
        ----------
        dic: dict
            float to list objects, with potentially many repetitions
        quantile: float

        Returns
        -------
        dict
            float to float
        """
        df = pd.DataFrame(dic.items(), columns=["keys", "values"])
        df["id"] = df["values"].apply(id)
        df_by_id = df.groupby("id")["values"].first().reset_index()
        df_by_id["quantiles"] = df_by_id["values"].apply(
            lambda l: np.percentile(l, quantile * 100)
        )
        df_by_id = df_by_id[["id", "quantiles"]].merge(df, on="id")
        return dict(zip(df_by_id["keys"], df_by_id["quantiles"]))

    def predict(self, x_test, quantile: float) -> List:
        """
        Quantile prediction.

        Parameters
        ----------
        x_test: list of lists
        quantile

        Returns
        -------
        list
            list of floats
        """
        predicted_values = []
        if quantile in self.quantile_dicts:
            quantile_dic = self.quantile_dicts[quantile]
        else:
            quantile_dic = self._get_quantiles_from_dic_with_list_values(
                self.cell_values_dict, quantile
            )
            self.quantile_dicts[quantile] = quantile_dic
        # Remember dic per quantile and use if already done
        for pt in x_test:
            pred = self.model.predict(np.array([pt]))[
                0
            ]  # xgboost doesn't like
            # lists
            closest_pred = self.get_closest_pt(self.cell_values, pred)
            predicted_values.append(quantile_dic[closest_pred])
        return predicted_values

    def estimate_dist(self, x_test: List[List[float]]) -> List:
        """
        Get estimate of sampling of Y|X=x for each x in x_test

        Parameters
        ----------
        x_test

        Returns
        -------
        list
            list of lists
        """
        predicted_samples = []
        for pt in x_test:
            pred = self.model.predict(np.array([pt]))[0]
            closest_pred = self.get_closest_pt(self.cell_values, pred)
            predicted_samples.append(self.cell_values_dict[closest_pred])
        return predicted_samples

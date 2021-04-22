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

from typing import Dict, List, Optional, Union

import copy
import numpy as np
import pandas as pd
import xgboost
import gc
from collections import defaultdict

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
            self.model = copy.deepcopy(model)
        else:
            self.model = self._create_xgboost_model(xgboost_params)
        self.clump_size = clump_size
        self.sorted_train_preds = None
        self.x_train_is_dataframe = None
        self.id_to_bins = None
        self.preds_to_id = None
        self.quantile_dicts = defaultdict(dict)

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

    def fit(
        self,
        x_train: Union[pd.DataFrame, List],
        y_train: Union[pd.Series, List],
        max_sample_size: Optional[
            int
        ] = None,  # If not None, choose without replacement
        # replacement min(max_sample_size, len(x_train)) many datapoints
        # to train on.
        seed: int = 1,
        x_train_is_dataframe: bool = False,  # This should be False for
        # XGBoost, but True if one uses lightgbm.
        **kwargs
    ):
        """
        Fits self.model and partitions R^n into cells. More accurately,
        it creates two dictionaries: self.preds_to_ids whose keys are the
        predictions of the training dataset and whose values are the ids of
        their associated bins, and self.ids_to_bins whose keys are the ids
        of the bins and whose values are associated lists of true values.
        """
        self.x_train_is_dataframe = x_train_is_dataframe
        self.quantile_dicts = defaultdict(dict)
        if not x_train_is_dataframe:
            x_train, y_train = np.array(x_train), np.array(y_train)  # xgboost
        # doens't like lists
        if max_sample_size and x_train_is_dataframe:
            assert max_sample_size > 0
            sample_size = min(max_sample_size, len(x_train))
            x_train = x_train.sample(
                n=min(sample_size, len(x_train)),
                replace=False,
                random_state=seed,
            )
            y_train = y_train[x_train.index]
        elif max_sample_size:
            assert max_sample_size > 0
            sample_size = min(max_sample_size, len(x_train))
            np.random.seed(seed)
            idx = np.random.choice(
                np.arange(len(x_train)), sample_size, replace=False
            )
            x_train = x_train[idx]
            y_train = y_train[idx]
        self.model.fit(x_train, y_train, **kwargs)
        y_train_pred = self.model.predict(x_train)
        df = pd.DataFrame(
            {
                "y_true": y_train,
                "y_pred": y_train_pred,
            }
        ).reset_index(drop=True)
        self.sorted_train_preds = sorted(df["y_pred"].unique())
        cell_values_dict = self.preprocess_df(df, clump_size=self.clump_size)
        del df
        gc.collect()
        cell_values_dict_df = pd.DataFrame(
            cell_values_dict.items(), columns=["keys", "values"]
        )
        cell_values_dict_df["id"] = cell_values_dict_df["values"].apply(id)
        self.id_to_bins = (
            cell_values_dict_df.groupby("id")["values"].first().to_dict()
        )
        self.preds_to_id = (
            cell_values_dict_df.groupby("keys")["id"].first().to_dict()
        )
        del cell_values_dict_df
        del cell_values_dict
        gc.collect()

    @staticmethod
    def clump(
        dic: Dict, min_num: int, sorted_keys: Optional[List] = None
    ) -> Dict:
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
        sorted_keys: list
            sorted(dic.keys()) or None

        Returns
        -------
        dict
            float to list; with the values often having the same list object
            appear multiple times
        """
        if sorted_keys is None:
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

    def preprocess_df(self, df: pd.DataFrame, clump_size: int = 100) -> Dict:
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
        dic = self.clump(dic, clump_size, self.sorted_train_preds)
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

    def _get_and_cache_quantile_computation(
        self, feature_vector_in_train: List, quantile: float
    ):
        """
        Updates self.quantile_dicts[quantile][feature_vector_in_train] to be the quantile of the associated true value bin.

        Parameters
        ----------
        feature_vector_in_train: list
             Feature vector that appears in the training data.
        quantile: float

        Returns
        -------
        float
            The quantile of the associated true value bin.
        """
        if feature_vector_in_train not in self.quantile_dicts[quantile]:
            self.quantile_dicts[quantile][
                feature_vector_in_train
            ] = np.percentile(
                self.id_to_bins[self.preds_to_id[feature_vector_in_train]],
                quantile * 100,
            )
        return self.quantile_dicts[quantile][feature_vector_in_train]

    def predict(
        self, x_test: Union[pd.DataFrame, List], quantile: float
    ) -> List:
        """
        Quantile prediction.

        Parameters
        ----------
        x_test: pd.DataFrame if self.x_train_is_dataframe, else list of
        lists
        quantile: float

        Returns
        -------
        list
            list of floats
        """
        if self.x_train_is_dataframe:
            preds = self.model.predict(x_test)
            predicted_values = [
                self._get_and_cache_quantile_computation(
                    self.get_closest_pt(self.sorted_train_preds, pred),
                    quantile,
                )
                for pred in preds
            ]
        else:
            predicted_values = []
            for pt in x_test:
                pred = self.model.predict(np.array([pt]))[
                    0
                ]  # xgboost doesn't like lists
                closest_pred = self.get_closest_pt(
                    self.sorted_train_preds, pred
                )
                predicted_values.append(
                    self._get_and_cache_quantile_computation(
                        closest_pred, quantile
                    )
                )
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
            closest_pred = self.get_closest_pt(self.sorted_train_preds, pred)
            predicted_samples.append(
                self.id_to_bins[self.preds_to_id[closest_pred]]
            )
        return predicted_samples

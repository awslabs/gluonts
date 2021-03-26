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

import glob
import json
from typing import Dict, List, Tuple

import mxnet as mx
import numpy as np

from gluonts.core.component import validated

EPOCH_INFO_STRING = "epoch-info"


def save_epoch_info(tmp_path: str, epoch_info: dict) -> None:
    r"""
    Writes the current epoch information into a json file in the model path.

    Parameters
    ----------
    tmp_path
        Temporary base path to save the epoch info.
    epoch_info
        Epoch information dictionary containing the parameters path, the epoch number and the tracking metric value.

    Returns
    -------
    None
    """

    with open("{}-{}.json".format(tmp_path, EPOCH_INFO_STRING), "w") as f:
        json.dump(epoch_info, f)


class AveragingStrategy:
    @validated()
    def __init__(
        self,
        num_models: int = 5,
        metric: str = "score",
        maximize: bool = False,
    ):
        r"""
        Parameters
        ----------
        num_models
            Number of model checkpoints to average.
        metric
            Metric which is used to average models.
        maximize
            Boolean flag to indicate whether the metric should be maximized or minimized.
        """
        self.num_models = num_models
        self.metric = metric
        self.maximize = maximize

    def apply(self, model_path: str) -> str:
        r"""
        Averages model parameters of serialized models based on the selected model strategy and metric.
        IMPORTANT: Depending on the metric the user might want to minimize or maximize. The maximize flag has to be
        chosen appropriately to reflect this.

        Parameters
        ----------
        model_path
            Path to the models directory.

        Returns
        -------
        Path to file with the averaged model.
        """
        checkpoints = self.get_checkpoint_information(model_path)

        checkpoint_paths, weights = self.select_checkpoints(checkpoints)

        average_parms = self.average(checkpoint_paths, weights)

        average_parms_path = model_path + "/averaged_model-0000.params"
        mx.nd.save(average_parms_path, average_parms)
        return average_parms_path

    @staticmethod
    def get_checkpoint_information(model_path: str) -> List[Dict]:
        r"""

        Parameters
        ----------
        model_path
            Path to the models directory.

        Returns
        -------
        List of checkpoint information dictionaries (metric, epoch_no, checkpoint path).
        """
        epoch_info_files = glob.glob(
            "{}/*-{}.json".format(model_path, EPOCH_INFO_STRING)
        )

        assert (
            len(epoch_info_files) >= 1
        ), "No checkpoints found in {}.".format(model_path)

        all_checkpoint_info = list()
        for epoch_info in epoch_info_files:
            with open(epoch_info) as f:
                all_checkpoint_info.append(json.load(f))
        return all_checkpoint_info

    def select_checkpoints(
        self, checkpoints: List[Dict]
    ) -> Tuple[List[str], List[float]]:
        r"""
        Selects checkpoints and computes weights for the selected checkpoints.

        Parameters
        ----------
        checkpoints
            List of checkpoint information dictionaries.
        Returns
        -------
            List of selected checkpoint paths and list of corresponding weights.
        """
        raise NotImplementedError()

    def average(self, param_paths: List[str], weights: List[float]) -> Dict:
        r"""
        Averages parameters from a list of .params file paths.

        Parameters
        ----------
        param_paths
            List of paths to parameter files.
        weights
            List of weights for the parameter average.

        Returns
        -------
        Averaged parameter dictionary.
        """
        all_arg_params = []

        for path in param_paths:
            params = mx.nd.load(path)
            all_arg_params.append(params)

        avg_params = {}
        for k in all_arg_params[0]:
            arrays = [p[k] for p in all_arg_params]
            avg_params[k] = self.average_arrays(arrays, weights)
        return avg_params

    @staticmethod
    def average_arrays(
        arrays: List[mx.nd.NDArray], weights: List[float]
    ) -> mx.nd.NDArray:
        r"""
        Takes a list of arrays of the same shape and computes the element wise weighted average.

        Parameters
        ----------
        arrays
            List of NDArrays with the same shape that will be averaged.
        weights
            List of weights for the parameter average.

        Returns
        -------
        The average of the NDArrays in the same context as arrays[0].
        """

        def _assert_shapes(arrays):
            shape_set = set([array.shape for array in arrays])
            assert (
                len(shape_set) == 1
            ), "All arrays should be the same shape. Found arrays with these shapes instead :{}".format(
                shape_set
            )

        _assert_shapes(arrays)

        if not arrays:
            raise ValueError("arrays is empty.")
        if len(arrays) == 1:
            return arrays[0]
        return mx.nd.add_n(*[a * w for a, w in zip(arrays, weights)])


class SelectNBestSoftmax(AveragingStrategy):
    def select_checkpoints(
        self, checkpoints: List[Dict]
    ) -> Tuple[List[str], List[float]]:
        r"""
        Selects the checkpoints with the best metric values.
        The weights are the softmax of the metric values, i.e.,
        w_i = exp(v_i) / sum(exp(v_j)) if maximize=True
        w_i = exp(-v_i) / sum(exp(-v_j)) if maximize=False

        Parameters
        ----------
        checkpoints
            List of checkpoint information dictionaries.
        Returns
        -------
            List of selected checkpoint paths and list of corresponding weights.
        """

        metric_path_tuple = [
            (c[self.metric], c["params_path"]) for c in checkpoints
        ]
        top_checkpoints = sorted(metric_path_tuple, reverse=self.maximize)[
            : self.num_models
        ]

        # weights of top checkpoints
        weights = [
            np.exp(c[0]) if self.maximize else np.exp(-c[0])
            for c in top_checkpoints
        ]
        weights = [x / sum(weights) for x in weights]

        # paths of top checkpoints
        checkpoint_paths = [c[1] for c in top_checkpoints]

        return checkpoint_paths, weights


class SelectNBestMean(AveragingStrategy):
    def select_checkpoints(
        self, checkpoints: List[Dict]
    ) -> Tuple[List[str], List[float]]:
        r"""
        Selects the checkpoints with the best metric values.
        The weights are equal for all checkpoints, i.e., w_i = 1/N.

        Parameters
        ----------
        checkpoints
            List of checkpoint information dictionaries.
        Returns
        -------
            List of selected checkpoint paths and list of corresponding weights.
        """

        metric_path_tuple = [
            (c[self.metric], c["params_path"]) for c in checkpoints
        ]
        top_checkpoints = sorted(metric_path_tuple, reverse=self.maximize)[
            : self.num_models
        ]

        # weights of top checkpoints
        weights = [1 / len(top_checkpoints)] * len(top_checkpoints)

        # paths of top checkpoints
        checkpoint_paths = [c[1] for c in top_checkpoints]

        return checkpoint_paths, weights

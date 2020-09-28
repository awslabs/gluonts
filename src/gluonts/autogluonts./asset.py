import mxnet as mx
from gluonts.trainer import learning_rate_scheduler as lrs
from estimator import net
from gluonts.support.util import get_hybrid_forward_input_names
import mxnet.gluon.nn as nn
batch_size = 32
def loss_value(loss) -> float:
    return loss.get_name_value()[0][1]
halt = False
input_names=get_hybrid_forward_input_names(net)


def count_model_params(self, net: nn.HybridBlock) -> int:
        params = net.collect_params()
        num_params = 0
        for p in params:
            v = params[p]
            num_params += np.prod(v.shape)
        return num_params


lr_scheduler = lrs.MetricAttentiveScheduler(
    objective="min",
    patience=10,
    decay_factor=0.5,
    min_lr=5e-5,
)

optimizer = mx.optimizer.Adam(
    learning_rate=1e-3,
    lr_scheduler=lr_scheduler,
    wd=1e-8,
    clip_gradient=10.0,
)

import mxnet as mx



import glob
import json

# Standard library imports
from typing import Any, Dict, List, Tuple

import mxnet as mx

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated

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

class IterationAveragingStrategy:

    r"""
    The model averaging is based on paper
    "Stochastic Gradient Descent for Non-smooth Optimization: Convergence Results and Optimal Averaging Schemes",
    (http://proceedings.mlr.press/v28/shamir13.pdf),
    which implements polynomial-decay averaging, parameterized by eta.
    When eta = 0, it is equivalent to simple average over all iterations with same weights.
    """



    @validated()
    def __init__(self, eta: float = 0):
        r"""
        Parameters
        ----------
        eta
            Parameter of polynomial-decay averaging.
        """

        self.eta = eta
        # Dict that maintains the averaged model parameters.
        self.averaged_model = None
        # Temporarily save the current model, so that the averaged model can be used for validation.
        self.cached_model = None
        # The number of models accumulated in the average.
        self.average_counter = 0
        # Indicate whether the model averaging has started.
        self.averaging_started = False

    def update_average_trigger(
        self, metric: Any = None, epoch: int = 0, **kwargs
    ):
        r"""
        Parameters
        ----------
        metric
            The criteria to trigger averaging.
        epoch
            The epoch to start averaging.
        Returns
        -------
        """
        raise NotImplementedError()

    def apply(self, model):
        r"""
        Parameters
        ----------
        model
            The model of the current iteration.
        Returns
        -------
        The averaged model, None if the averaging hasn't started.
        """

        if self.averaging_started:
            self.update_average(model)

        return self.averaged_model

    def update_average(self, model):
        r"""
        Parameters
        ----------
        model
            The model to update the average.
        """
        self.average_counter += 1
        if self.averaged_model is None:
            self.averaged_model = {
                k: v.list_data()[0].copy()
                for k, v in model.collect_params().items()
            }
        else:
            alpha = (self.eta + 1.0) / (self.eta + self.average_counter)
            # moving average
            for name, param_avg in self.averaged_model.items():
                param_avg[:] += alpha * (
                    model.collect_params()[name].list_data()[0] - param_avg
                )

    def load_averaged_model(self, model):
        r"""
        When validating/evaluating the averaged model in the half way of training,
        use load_averaged_model first to load the averaged model and overwrite the current model,
        do the evaluation, and then use load_cached_model to load the current model back.
        Parameters
        ----------
        model
            The model that the averaged model is loaded to.
        """
        if self.averaged_model is not None:
            # cache the current model
            if self.cached_model is None:
                self.cached_model = {
                    k: v.list_data()[0].copy()
                    for k, v in model.collect_params().items()
                }
            else:
                for name, param_cached in self.cached_model.items():
                    param_cached[:] = model.collect_params()[name].list_data()[
                        0
                    ]
            # load the averaged model
            for name, param_avg in self.averaged_model.items():
                model.collect_params()[name].set_data(param_avg)

    def load_cached_model(self, model):
        r"""
        Parameters
        ----------
        model
            The model that the cached model is loaded to.
        """
        if self.cached_model is not None:
            # load the cached model
            for name, param_cached in self.cached_model.items():
                model.collect_params()[name].set_data(param_cached)


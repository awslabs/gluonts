# Standard library imports
import glob
import json

# Third-party imports
import mxnet as mx
import numpy as np

EPOCH_INFO_STRING = "epoch-info"


def save_epoch_info(tmp_path, epoch_info):
    """
    Writes the current epoch information into a json file in the model path.
    :param epoch_info: Dictionary, containing with epoch information.
    :return: None
    """
    with open("{}-{}.json".format(tmp_path, EPOCH_INFO_STRING), "w") as f:
        json.dump(epoch_info, f)


def average_parameters(
    model_path, num_models=5, metric="score", maximize=False, weight="average"
):
    """
    Averages model parameters of serialized models based on the selected model strategy and metric.
    IMPORTANT: Depending on the metric the user might want to minimize or maximize. The maximize flag has to be
    chosen appropriately to reflect this.

    :param model_path: str, path to the models directory.
    :param num_models: int, number of model checkpoints to average.
    :param metric: str, metric which is used to average models.
    :param maximize: boolean, flag to indicate whether the metric should be maximized or minimized (important for some
                     strategies, such as best).
    :param weight: str, weights for the parameter average.
    :return: str, path to file with averaged model
    """
    checkpoints = load_checkpoints(model_path, num_models, metric, maximize)
    checkpoint_paths = [checkpoint[1] for checkpoint in checkpoints]

    if weight == "average":
        weights = [1 / len(checkpoints)] * len(checkpoints)
    elif weight == "exp-metric":
        weights = [
            np.exp(checkpoint[0]) if maximize else np.exp(-checkpoint[0])
            for checkpoint in checkpoints
        ]
        weights = [x / sum(weights) for x in weights]
    else:
        raise ValueError("Unknown value for 'weight'.")

    average_parms = average(checkpoint_paths, weights)

    average_parms_path = model_path + "/averaged_model-0000.params"
    mx.nd.save(average_parms_path, average_parms)
    return average_parms_path


def load_checkpoints(model_path, num_models=5, metric="score", maximize=False):
    """
    Load checkpoints with serialized model information.

    :param model_path: str, path to the models directory.
    :param num_models: int, number of model checkpoints to average.
    :param metric: str, metric which is used to average models.
    :param maximize: boolean, flag to indicate whether the metric should be maximized or minimized (important for some
                     strategies, such as best).
    :return: top checkpoints, selected by chosen strategy.
    """
    all_checkpoint_info = get_checkpoint_information(model_path)

    checkpoints = [
        (checkpoint_info[metric], checkpoint_info["params_path"])
        for checkpoint_info in all_checkpoint_info
    ]
    top_checkpoints = _strategy_best(checkpoints, num_models, maximize)

    return top_checkpoints


def get_checkpoint_information(model_path):
    """
    Loads checkpoint information from model path.

    :param model_path: str, path to model directory.
    :return: list of checkpoint information tuples (metric, checkpoint path).
    """
    epoch_info_files = glob.glob(
        "{}/*-{}.json".format(model_path, EPOCH_INFO_STRING)
    )

    assert len(epoch_info_files) >= 1, "No checkpoints found in {}.".format(
        model_path
    )

    all_checkpoint_info = list()
    for epoch_info in epoch_info_files:
        with open(epoch_info) as f:
            all_checkpoint_info.append(json.load(f))
    return all_checkpoint_info


def average(param_paths, weight):
    """
    Averages parameters from a list of .params file paths.
    :param param_paths: List of paths to parameter files.
    :param weight: str, weights for the parameter average.
    :return: Averaged parameter dictionary.
    """
    all_arg_params = []
    # all_aux_params = []

    for path in param_paths:
        params = mx.nd.load(path)
        all_arg_params.append(params)

    avg_params = {}
    for k in all_arg_params[0]:
        arrays = [p[k] for p in all_arg_params]
        avg_params[k] = average_arrays(arrays, weight)
    return avg_params


def average_arrays(arrays, weight):
    """
    Take a list of arrays of the same shape and take the element wise average.
    :param arrays: A list of NDArrays with the same shape that will be averaged.
    :param weight: str, weights for the parameter average.
    :return: The average of the NDArrays in the same context as arrays[0].
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
    return mx.nd.add_n(*[a * w for a, w in zip(arrays, weight)])


def _strategy_best(checkpoints, num_models, maximize):
    top_n = sorted(checkpoints, reverse=maximize)[:num_models]
    return top_n

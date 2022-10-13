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
import argparse
import yaml
import os
import wandb
import shutil
import torch
import numpy as np


def str2bool(v):
    """
    Source code copied from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_dict_from_yaml(file_path):
    """
    Load args from .yml file.
    """
    with open(file_path) as file:
        yaml_dict = yaml.safe_load(file)

    # create argsparse object
    # parser = argparse.ArgumentParser(description='MFCVAE training')
    # args, unknown = parser.parse_known_args()
    # for key, value in config.items():
    #     setattr(args, key, value)

    # return args.__dict__  # return as dict, not as argsparse
    return yaml_dict


def download_some_wandb_files(files_to_restore, run_id):

    api = wandb.Api()
    wandb_args = load_dict_from_yaml("setup/wandb.yml")
    run_path = wandb_args["team_name"] + "/" + wandb_args["project_name"] + "/" + run_id
    run = api.run(run_path)
    download_dir = wandb.run.dir

    os.chdir(download_dir)

    print("Download directory: ", os.getcwd())

    # restore the files files
    for file_path in files_to_restore:
        # run.file(file_path).download()   # is an alternative, should also work
        wandb.restore(file_path, run_path=run_path)

    os.chdir("../../../")


def check_hyperparams(H):
    # various checks that enc_spec/dec_spec the string is well-behaved 1) only powers of two in down and up etc.
    pass

    # if H['symmetric_mode']:
    #     if H['last_from_enc_shared_stack']:
    #         # must have same number of stacks in encoder and decoder per resolution
    #         count_stacks_enc, count_stacks_dec = {}, {}
    #         blockstrings = H['enc_blocks'].split(",")
    #         # initialise all reasonable keys with 0
    #         max_res = int(blockstrings[0].split('x')[0] if 'x' in blockstrings[0] else blockstrings[0].split('r')[0])
    #         for res in range(max_res+1):
    #             count_stacks_enc[res], count_stacks_dec[res] = 0, 0
    #         # count stacks per resolution in encoder
    #         for ss in blockstrings:
    #             if 'x' in ss:
    #                 res = int(ss.split('x')[0])
    #                 count_stacks_enc[res] += 1
    #             elif 'r' in ss:
    #                 res = int(ss.split('r')[0])
    #                 count_stacks_enc[res] += 1
    #             # other cases ignored for just counting stacks
    #
    #         blockstrings = H['dec_blocks'].split(",")
    #         for ss in blockstrings:
    #             if 'x' in ss:
    #                 res = int(ss.split('x')[0])
    #                 count_stacks_dec[res] += 1
    #             elif 'r' in ss:
    #                 res = int(ss.split('r')[0])
    #                 count_stacks_dec[res] += 1
    #             # other cases ignored for just counting stacks
    #
    #
    #         # check if counts match
    #         for res in range(max_res+1):
    #             if count_stacks_enc[res] != count_stacks_dec[res]:
    #                 exit("Encoder and Decoder must have same number of stacks per resolution (violated in res %d)!"%(res))
    #
    #     else:
    #         # must have same number of blocks PER RESOLUTION in encoder and decoder
    #         count_blocks_enc, count_blocks_dec = {}, {}
    #         blockstrings = H['enc_blocks'].split(",")
    #         # initialise all reasonable keys with 0
    #         max_res = int(blockstrings[0].split('x')[0] if 'x' in blockstrings[0] else blockstrings[0].split('r')[0])
    #         for i in range(max_res+1):
    #             count_blocks_enc[i], count_blocks_dec[i] = 0, 0
    #         for i, ss in enumerate(blockstrings):
    #             if "x" in ss:
    #                 res, count = ss.split("x")
    #                 count_blocks_enc[int(res)] += int(count)
    #             elif "r" in ss:
    #                 res, count = ss.split("r")
    #                 count_blocks_enc[int(res)] += int(count)
    #             elif 'd' in ss:
    #                 next_string = blockstrings[i+1]
    #                 res = int(next_string.split('x')[0]) if 'x' in next_string else int(next_string.split('r')[0])
    #                 count_blocks_enc[res] += 1
    #             else:
    #                 exit("unparsed string.")
    #
    #         blockstrings = H['dec_blocks'].split(",")
    #         for i, ss in enumerate(blockstrings):
    #             if "x" in ss:
    #                 res, count = ss.split("x")
    #                 count_blocks_dec[int(res)] += int(count)
    #             elif "r" in ss:
    #                 res, count = ss.split("r")
    #                 count_blocks_dec[int(res)] += int(count)
    #             elif 'm' in ss:
    #                 next_string = blockstrings[i+1]
    #                 res = int(next_string.split('x')[0]) if 'x' in next_string else int(next_string.split('r')[0])
    #                 count_blocks_dec[res] += 1
    #             else:
    #                 exit("unparsed string.")
    #
    #         for res in range(max_res+1):
    #             if count_blocks_enc[res] != count_blocks_dec[res]:
    #                 exit("Encoder and Decoder must have same number of blocks per resolution (violated in res %d; note that first block in enc and dec must be +1)!"%(res))


def parse_dec_spec_list(enc_spec, dec_spec, input_resolution):
    res = int(
        compute_bottleneck_res(data_resolution=input_resolution, enc_spec=enc_spec)
    )
    dec_spec_split = dec_spec.split(",")
    block_label_list = []
    for s in dec_spec_split:
        if "u" in s:
            up_rate = int(s[1:])  # cut away 'u' and interpret as int
            res = int(res * up_rate)
            # Note: could also look at 'u' layers in addition, but not done here because various changes would be necessary
            # block_label_list += [str(res) + 'x' + str(res) + ' (up)']
        elif "r" in s:
            s_split = s.split("r")
            n_blocks, n_reps_per_block = int(s_split[0]), int(s_split[1])
            for _ in range(n_blocks):
                block_label_list += [str(res) + "x" + str(res)]
                for _ in range(n_reps_per_block - 1):
                    block_label_list += [str(res) + "x" + str(res) + " (rep)"]
        else:
            for _ in range(int(s)):
                block_label_list += [str(res) + "x" + str(res)]

    return block_label_list


def compute_gradient_norm(parameters, norm_type=2.0):
    """
    taken from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch._six.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )

    return total_norm


def get_sample_for_visualization(generator, normalize_fn, device):
    # print("in get sample vis")
    # get a single batch
    batch = next(generator)
    x_context, x_forecast = batch["past_target"], batch["future_target"]
    x_item_id = batch["item_id"]
    if "cuda" in device:
        x_context = x_context.cuda(non_blocking=True)
        x_forecast = x_forecast.cuda(non_blocking=True)

    x_context = insert_channel_dim(x_context)
    x_forecast = insert_channel_dim(x_forecast)

    x_context_orig, x_forecast_orig = x_context.clone(), x_forecast.clone()

    x_context_input = normalize_fn(x_context, ids=x_item_id)
    x_forecast_input = normalize_fn(x_forecast, ids=x_item_id)

    return x_context_orig, x_forecast_orig, x_context_input, x_forecast_input, x_item_id


def compute_stochastic_depth(dec_spec):
    dec_spec_split = dec_spec.split(",")
    stoch_depth = 0
    for j, s in enumerate(dec_spec_split):
        if "u" in s:
            continue  # not a layer, just continue
        elif "r" in s:
            s_split = s.split("r")
            n_blocks, n_reps_per_block = int(s_split[0]), int(s_split[1])
            stoch_depth += n_blocks * n_reps_per_block
        else:
            stoch_depth += int(s)

    return stoch_depth


def compute_bottleneck_res(data_resolution, enc_spec=None, dec_spec=None):
    res = data_resolution

    if enc_spec is not None:
        enc_spec_split = enc_spec.split(",")
        for j, s in enumerate(enc_spec_split):
            if "d" in s:
                down_rate = int(s[1:])
                res /= down_rate
    else:
        dec_spec_split = dec_spec.split(",")
        for j, s in enumerate(dec_spec_split):
            if "u" in s:
                up_rate = int(s[1:])
                res /= up_rate

    return res


def compute_blocks_per_res(spec, enc_or_dec, input_resolution, count_up_down=False):
    res_to_n_layers = {}
    spec_split = spec.split(",")

    # find res of first layer in the spec
    if enc_or_dec == "enc":
        res = input_resolution
    elif enc_or_dec == "dec":
        # find the resolution in the bottleneck from decoder (since spec is decoder!) --> can't use compute_bottleneck_res
        u_spec = [s for s in spec_split if "u" in s]
        u_spec = reversed(u_spec)
        res = input_resolution
        for u in u_spec:
            up_rate = int(u[1:])  # cut 'u' off and interpret as integer
            res /= up_rate
    else:
        raise Exception()

    for j, s in enumerate(spec_split):

        if res not in res_to_n_layers.keys():
            res_to_n_layers[res] = 0

        if "u" in s:
            res = int(res * int(s[1:]))
            if res not in res_to_n_layers.keys():
                res_to_n_layers[res] = 0
            if count_up_down:
                res_to_n_layers[res] += 1
        elif "d" in s:
            res = int(res / int(s[1:]))
            if res not in res_to_n_layers.keys():
                res_to_n_layers[res] = 0
            if count_up_down:
                res_to_n_layers[res] += 1
        elif "r" in s:
            s_split = s.split("r")
            res_to_n_layers[res] += int(int(s_split[0]) * int(s_split[1]))
        else:
            res_to_n_layers[res] += int(s)

    return res_to_n_layers


def append_to_list_(*args):
    # append to list
    for arg in args:
        arg[1].append(arg[0])
    # just return lists
    lists = [arg[1] for arg in args]
    # no return value, because happens in place


def finites_only(*args):
    """

    :param args: multiple arguments, each a list of values
    :return: a tuple of the multiple arguments. for each argument, non-finite values in the list have been removed and the list has been converted to a numpy array.
    """
    finites = []
    for arg in args:
        arg = np.array(arg)  # convert outer most to numpy array
        arg = arg[np.isfinite(arg)]
        # arg = arg.tolist()  # convert back to list
        finites.append(arg)

    return tuple(finites)


def insert_channel_dim(x):
    if len(x.shape) == 2:
        x = torch.unsqueeze(x, 1)

    return x


def calculate_padding_base_2(H):
    pad_context = nextPowerOf2(H.context_length) - H.context_length
    pad_forecast = nextPowerOf2(H.forecast_length) - H.forecast_length

    return pad_context, pad_forecast


# Compute power of two greater than or equal to `n`
def nextPowerOf2(n):
    count = 0

    # First n in the below
    # condition is for the
    # case where n is 0
    if n and not (n & (n - 1)):
        return n

    while n != 0:
        n >>= 1
        count += 1

    return 1 << count


def get_stable_scale(log_sigma, constant=1e-6):
    return torch.exp(log_sigma) + constant  # 0.5 *


def gaussian_analytical_kl(mu1, mu2, log_sigma1, log_sigma2):
    """
    KL[p_1(m0,log_sigma0), p_2(mu1, log_sigma1)]
    """
    log_var_ratio = 2 * (log_sigma1 - log_sigma2)
    t1 = (mu1 - mu2) ** 2 / (2 * log_sigma2).exp()
    sum_dims = list(range(len(mu1.size())))[1:]  # all except batch dimension
    return (0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)).sum(dim=sum_dims)


if __name__ == "__main__":
    # unit test
    # wandb_config = load_dict_from_yaml('setup/wandb.yml')
    # print(wandb_config)

    # unit test
    # a = ['a']
    # c = ['c']
    # print(append_to_list_(('b', a), ('d', c)))
    # print(a)
    # print(c)

    # unit test
    # a = np.array([np.nan, 1.])
    # b = np.array([2., np.inf])
    # c, d = finites_only(a, b)
    # print(c, d)

    # unit test
    # print(compute_bottleneck_res(input_resolution=32, enc_spec="2,d2,3r3"))

    # unit test
    # print(compute_latents_per_res(spec='3,d2,4r3', enc_or_dec='enc', input_resolution=32))

    # unit test
    print(compute_stochastic_depth(dec_spec="2,u2,3r3"))

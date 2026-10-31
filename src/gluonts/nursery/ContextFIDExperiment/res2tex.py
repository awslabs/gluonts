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

import json
import pathlib
import re
from pathlib import Path

import pandas as pd
import torch


class toLatex:
    def __init__(
        self,
        path_to_config_file_gan: pathlib.PosixPath() = None,
        path_to_config_file_CNN: pathlib.PosixPath() = None,
        path_to_pred_net_timegan: pathlib.PosixPath() = None,
        path_to_pred_net_ebgan: pathlib.PosixPath() = None,
    ):
        if path_to_config_file_gan:
            self.path_to_config_file_gan = path_to_config_file_gan
            textfile = open(self.path_to_config_file_gan, "r")
            self.filetext = textfile.read()
            textfile.close()

        if path_to_config_file_CNN:
            self.path_to_config_file_CNN = path_to_config_file_CNN
            textfile_CNN = open(self.path_to_config_file_CNN, "r")
            self.filetext_CNN = textfile_CNN.read()
            textfile_CNN.close()

        if path_to_pred_net_timegan:
            self.path_to_pred_net_timegan = path_to_pred_net_timegan
            f = open(Path(path_to_pred_net_timegan) / "prediction_net.json")
            self.timegan_data = json.load(f)
            f.close()
            f = open(Path(path_to_pred_net_timegan) / "parameters.json")
            self.timegan_parameters = json.load(f)
            f.close()

        if path_to_pred_net_ebgan:
            self.path_to_pred_net_ebgan = path_to_pred_net_ebgan
            f = open(path_to_pred_net_ebgan + "/prediction_net.json")
            self.ebgan_data = json.load(f)
            f.close()
            f = open(path_to_pred_net_ebgan + "/parameters.json")
            self.ebgan_parameters = json.load(f)
            f.close()

    def _str2bool(self, v):
        return v.lower() in ("true")

    def _str2float(self, v):
        return tuple(float(s) for s in v)

    def _str2int(self, v):
        return tuple(int(s) for s in v)

    def _get_momment_loss(self):
        pattern = re.compile(r"momment_loss=(\S+),")
        momment_loss = pattern.findall(self.filetext)[0]
        return self._str2bool(momment_loss)

    def _get_use_loss(self):
        pattern = re.compile(r'use_loss="(\S+)"\),')
        use_loss = pattern.findall(self.filetext)[0]
        return use_loss

    def _get_num_epochs(self):
        pattern = re.compile(r"num_epochs=(\S+),")
        num_epochs = pattern.findall(self.filetext)[0]
        return int(num_epochs)

    def _get_batch_size(self):
        pattern = re.compile(r"batch_size=(\S+),")
        batch_size = pattern.findall(self.filetext)[0]
        return int(batch_size)

    def _get_path_to_pretrain(self):
        pattern = re.compile(r'path_to_pretrain="(\S+)",')
        path_to_pretrain = pattern.findall(self.filetext)[0].split("/")
        try:
            model = path_to_pretrain[-2]
            epoch = path_to_pretrain[-1].split("_")[-1][:-3]
            return model + "/" + epoch
        except Exception:
            return "No pre-train"

    def _get_target_len(self):
        pattern = re.compile(r"target_len=(\d+)")
        target_len = pattern.findall(self.filetext)[0]
        return int(target_len)

    def _get_betas(self):
        pattern = re.compile(r"betas: \((\S+), (\S+)\)")
        betas = pattern.findall(self.filetext)
        betas_generator, betas_discriminator = betas[0], betas[1]
        return self._str2float(betas_generator), self._str2float(
            betas_discriminator
        )

    def _get_lr_discriminator(self):
        pattern = re.compile(r"lr_discriminator=(\S+),")
        lr_discriminator = pattern.findall(self.filetext)[0]
        return float(lr_discriminator)

    def _get_lr_generator(self):
        pattern = re.compile(r"lr_generator=(\S+),")
        lr_generator = pattern.findall(self.filetext)[0]
        return float(lr_generator)

    def _get_nb_step_discrim(self):
        pattern = re.compile(r"nb_step_discrim=(\S+),")
        nb_step_discrim = pattern.findall(self.filetext)[0]
        return int(nb_step_discrim)

    def _get_schedule(self):
        pattern = re.compile(r"(?<=Pre-train schedule : )\[.*?\]")
        schedule = pattern.findall(self.filetext)[0]
        return schedule

    def _get_nb_features(self):
        pattern = re.compile(r"nb_features=(\S+),")
        nb_features = pattern.findall(self.filetext)[0]
        return int(nb_features)

    def _get_ks_conv(self):
        pattern = re.compile(r"ks_conv=(\S+),")
        ks_conv = pattern.findall(self.filetext)[0]
        return int(ks_conv)

    def _get_key_features(self):
        pattern = re.compile(r"key_features=(\S+),")
        key_features = pattern.findall(self.filetext)[0]
        return int(key_features)

    def _get_value_features(self):
        pattern = re.compile(r"value_features=(\S+)\),")
        value_features = pattern.findall(self.filetext)[0]
        return int(value_features)

    def _get_ks_key(self):
        pattern = re.compile(r"ks_key=(\S+),")
        ks_key = pattern.findall(self.filetext)[0]
        return int(ks_key)

    def _get_ks_query(self):
        pattern = re.compile(r"ks_query=(\S+),")
        ks_query = pattern.findall(self.filetext)[0]
        return int(ks_query)

    def _get_ks_value(self):
        pattern = re.compile(r"ks_value=(\S+),")
        ks_value = pattern.findall(self.filetext)[0]
        return int(ks_value)

    def _get_dataset_name(self):
        pattern = re.compile(r"Dataset = (\S+)")
        dataset_name = pattern.findall(self.filetext)[0]
        return dataset_name

    def _get_scaling(self):
        pattern = re.compile(r"scaling=\"(\S+)\",")
        try:
            scaling = pattern.findall(self.filetext)[0]
        except Exception:
            scaling = "local"
        return str(scaling)

    def _get_cardinality(self):
        pattern = re.compile(r"cardinality=\[(\S+)\]")
        try:
            cardinality = int(pattern.findall(self.filetext)[0])
        except Exception:
            cardinality = None

        return cardinality

    def _get_self_attention(self):
        pattern = re.compile(r"self_attention=(\S+),")
        try:
            self_attention = pattern.findall(self.filetext)[0]
            print(self_attention)
            if self_attention.lower() == "true":
                return True
            else:
                return False
        except Exception:
            self_attention = True

        return self_attention

    def _get_channels_CNN(self):
        pattern = re.compile(r"channels=(\S+),")
        channels = pattern.findall(self.filetext_CNN)[0]
        return int(channels)

    def _get_in_channels_CNN(self):
        pattern = re.compile(r"in_channels=(\S+),")
        in_channels = pattern.findall(self.filetext_CNN)[0]
        return int(in_channels)

    def _get_out_channels_CNN(self):
        pattern = re.compile(r"out_channels=(\S+),")
        out_channels = pattern.findall(self.filetext_CNN)[0]
        return int(out_channels)

    def _get_depth_CNN(self):
        pattern = re.compile(r"depth=(\S+),")
        depth = pattern.findall(self.filetext_CNN)[0]
        return int(depth)

    def _get_reduced_size_CNN(self):
        pattern = re.compile(r"reduced_size=(\S+),")
        reduced_size = pattern.findall(self.filetext_CNN)[0]
        return int(reduced_size)

    def _get_kernel_size_CNN(self):
        pattern = re.compile(r"kernel_size=(\S+),")
        kernel_size = pattern.findall(self.filetext_CNN)[0]
        return int(kernel_size)

    def _get_module_timegan(self):
        module = self.timegan_data["kwargs"]["generator"]["kwargs"]["module"]
        return module

    def _get_hidden_size_timegan(self):
        hidden_size = self.timegan_data["kwargs"]["generator"]["kwargs"][
            "hidden_size"
        ]
        return int(hidden_size)

    def _get_num_layers_timegan(self):
        num_layers = self.timegan_data["kwargs"]["generator"]["kwargs"][
            "num_layers"
        ]
        return int(num_layers)

    def _get_noise_size_timegan(self):
        noise_size = self.timegan_data["kwargs"]["generator"]["kwargs"][
            "noise_size"
        ]
        return int(noise_size)

    def _get_output_size_timegan(self):
        output_size = self.timegan_data["kwargs"]["recovery"]["kwargs"][
            "output_size"
        ]
        return int(output_size)

    def _get_which_network_timegan(self):
        which_network = self.timegan_data["kwargs"]["which_network"]
        return which_network

    def _get_prediction_length_timegan(self):
        prediction_length = self.timegan_parameters["prediction_length"]
        return int(prediction_length)

    def _get_batch_size_timegan(self):
        batch_size = self.timegan_parameters["batch_size"]
        return batch_size

    def _get_dataset_name_timegan(self):
        with open(
            Path(self.path_to_pred_net_timegan) / "dataset.txt", "r"
        ) as f:
            dataset_name = f.read()
        return dataset_name

    def _get_scaling_timegan(self):
        with open(
            Path(self.path_to_pred_net_timegan) / "scaling.txt", "r"
        ) as f:
            scaling = f.read()
        return scaling

    def _get_batch_size_ebgan(self):
        batch_size = self.ebgan_parameters["batch_size"]
        return batch_size

    def _get_prediction_length_ebgan(self):
        prediction_length = self.ebgan_parameters["prediction_length"]
        return int(prediction_length)

    def _get_cardinality_ebgan(self):
        cardinality = self.ebgan_data["kwargs"]["cardinality"][0]
        return cardinality

    def _get_embedding_dim_ebgan(self):
        embedding_dim = self.ebgan_data["kwargs"]["embedding_dim"]
        return embedding_dim

    def _get_hidden_dim_ebgan(self):
        hidden_dim = self.ebgan_data["kwargs"]["hidden_dim"]
        return hidden_dim

    def _get_input_dim_ebgan(self):
        input_dim = self.ebgan_data["kwargs"]["input_dim"]
        return input_dim

    def _get_out_dim_ebgan(self):
        out_dim = self.ebgan_data["kwargs"]["out_dim"]
        return out_dim

    def _get_real_embedding_dim_ebgan(self):
        real_embedding_dim = self.ebgan_data["kwargs"]["real_embedding_dim"]
        return real_embedding_dim

    def _get_rr_weight_ebgan(self):
        rr_weight = self.ebgan_data["kwargs"]["rr_weight"]
        return rr_weight

    def _get_z_dim_ebgan(self):
        z_dim = self.ebgan_data["kwargs"]["z_dim"]
        return z_dim

    def _get_dataset_name_ebgan(self):
        with open(Path(self.path_to_pred_net_ebgan) / "dataset.txt", "r") as f:
            dataset_name = f.read()
        return dataset_name

    def _get_scaling_ebgan(self):
        with open(Path(self.path_to_pred_net_ebgan) / "scaling.txt", "r") as f:
            scaling = f.read()
        return scaling

    def txt_to_dict(self):
        config_dict = {}
        config_dict["momment_loss"] = self._get_momment_loss()
        config_dict["use_loss"] = self._get_use_loss()
        config_dict["num_epochs"] = self._get_num_epochs()
        config_dict["batch_size"] = self._get_batch_size()
        config_dict["path_to_pretrain"] = self._get_path_to_pretrain()
        config_dict["target_len"] = self._get_target_len()
        (
            config_dict["betas_generator"],
            config_dict["betas_discriminator"],
        ) = self._get_betas()
        config_dict["lr_discriminator"] = self._get_lr_discriminator()
        config_dict["lr_generator"] = self._get_lr_generator()
        config_dict["nb_step_discrim"] = self._get_nb_step_discrim()
        config_dict["schedule"] = self._get_schedule()
        config_dict["dataset"] = self._get_dataset_name()
        config_dict["scaling"] = self._get_scaling()
        config_dict["cardinality"] = self._get_cardinality()
        config_dict["self_attention"] = self._get_self_attention()
        return config_dict

    def config_to_table(self, caption="My caption"):
        config = self.txt_to_dict()
        df = pd.DataFrame.from_dict(config, orient="index")
        latex = df.to_latex(index=True, header=False, caption=caption)
        return latex

    def CNN_txt_to_dict(self):
        config_dict = {}
        config_dict["channels"] = self._get_channels_CNN()
        config_dict["in_channels"] = self._get_in_channels_CNN()
        config_dict["out_channels"] = self._get_out_channels_CNN()
        config_dict["depth"] = self._get_depth_CNN()
        config_dict["reduced_size"] = self._get_reduced_size_CNN()
        config_dict["kernel_size"] = self._get_kernel_size_CNN()
        return config_dict

    def CNN_config_to_table(self, caption="My caption"):
        config = self.CNN_txt_to_dict()
        df = pd.DataFrame.from_dict(config, orient="index")
        latex = df.to_latex(index=True, header=False, caption=caption)
        return latex

    def TimeGan_txt_to_dict(self):
        config = {}
        config["module"] = self._get_module_timegan()
        config["hidden_size"] = self._get_hidden_size_timegan()
        config["num_layers"] = self._get_num_layers_timegan()
        config["noise_size"] = self._get_noise_size_timegan()
        config["output_size"] = self._get_output_size_timegan()
        config["which_network"] = self._get_which_network_timegan()
        config["prediction_length"] = self._get_prediction_length_timegan()
        config["batch_size"] = self._get_batch_size_timegan()
        try:
            config["dataset"] = self._get_dataset_name_timegan()
        except Exception:
            config["dataset"] = "Not specifed"

        try:
            config["scaling"] = self._get_scaling_timegan()
        except Exception:
            config["scaling"] = "Not specified"

        return config

    def TimeGan_config_to_table(self, caption="My caption"):
        config = self.TimeGan_txt_to_dict()
        df = pd.DataFrame.from_dict(config, orient="index")
        latex = df.to_latex(index=True, header=False, caption=caption)
        return latex

    def EBGAN_txt_to_dict(self):
        config = {}
        try:
            config["dataset"] = self._get_dataset_name_ebgan()
        except Exception:
            config["dataset"] = "Not specified"
        config["prediction_length"] = self._get_prediction_length_ebgan()
        config["cardinality"] = self._get_cardinality_ebgan()
        config["embedding_dim"] = self._get_embedding_dim_ebgan()
        config["hidden_dim"] = self._get_hidden_dim_ebgan()
        config["input_dim"] = self._get_input_dim_ebgan()
        config["out_dim"] = self._get_out_dim_ebgan()
        config["real_embedding"] = self._get_real_embedding_dim_ebgan()
        config["rr_weight"] = self._get_rr_weight_ebgan()
        config["z_dim"] = self._get_z_dim_ebgan()
        try:
            config["scaling"] = self._get_scaling_ebgan()
        except Exception:
            config["scaling"] = "Not specified"

        return config

    def EBGAN_config_to_table(self, caption="My caption"):
        config = self.EBGAN_txt_to_dict()
        df = pd.DataFrame.from_dict(config, orient="index")
        latex = df.to_latex(index=True, header=False, caption=caption)
        return latex

    def _mergedict(self, d1, d2):
        d1 = dict(d1._asdict())
        d2 = dict(d2._asdict())
        res = {
            k: str(round(d1.get(k, 0), 3))
            + r" pm "
            + str(round(d2.get(k, 0), 3))
            for k in set(d1)
        }
        return res

    def _result_to_dict(self, mean_res, std_res, abs_diff, std_abs_diff):
        res = self._mergedict(mean_res, std_res)
        res_diff = self._mergedict(abs_diff, std_abs_diff)
        return res, res_diff

    def _dict_to_latex(self, res_dict, caption: str, network: str = "MyGAN"):
        df = pd.DataFrame.from_dict(
            res_dict, orient="index", columns=[network]
        ).transpose()
        latex = df.to_latex(index=True, header=True, caption=caption)
        return latex

    def result_to_df(self, mean_res, std_res, abs_diff, std_abs_diff, network):
        def to_df(inp):
            di = dict(inp._asdict())
            df = pd.DataFrame.from_dict(
                di, orient="index", columns=[network]
            ).transpose()
            df.index.name = "Network"
            return df

        lis = [mean_res, std_res, abs_diff, std_abs_diff]
        return list(map(to_df, lis))

    def result_to_latex(
        self, mean_res, std_res, abs_diff, std_abs_diff, caption, network
    ):
        res_mean, res_diff = self._result_to_dict(
            mean_res, std_res, abs_diff, std_abs_diff
        )
        latex_mean = self._dict_to_latex(res_mean, caption, network)
        latex_diff = self._dict_to_latex(res_diff, caption, network)

        return latex_mean, latex_diff

    def _tensor_to_stat(self, t: torch.Tensor):
        return torch.mean(t).item(), torch.std(t).item()

    def _fid_to_dict(self, fid_ts: torch.Tensor, fid_embed: torch.Tensor):
        mean1, std1 = self._tensor_to_stat(fid_ts)
        mean2, std2 = self._tensor_to_stat(fid_embed)
        std_dict = {"fid_ts": std1, "fid_embed": std2}
        mean_dict = {"fid_ts": mean1, "fid_embed": mean2}

        return mean_dict, std_dict

    def fid_to_latex(
        self, fid_ts: torch.Tensor, fid_embed: torch.Tensor, caption, network
    ):
        mean_dict, std_dict = self._fid_to_dict(fid_ts, fid_embed)
        res = {
            k: str(round(mean_dict.get(k, 0), 3))
            + r" pm "
            + str(round(std_dict.get(k, 0), 3))
            for k in set(mean_dict)
        }
        latex = self._dict_to_latex(res, caption, network)
        return latex

    def _formating(self, d: dict):
        return str(round(d["mean"], 3)) + " pm " + str(round(d["std"], 3))

    def tsfresh_to_latex(self, ts_stat: dict, caption: str):
        df = pd.DataFrame(ts_stat).transpose().applymap(self._formating)
        return df.to_latex(caption=caption)

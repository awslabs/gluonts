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
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

df = pd.read_csv(Path(__file__).parent / "TrainingFollowUp.csv")


def filter_df(
    *,
    num_batches_per_epoch: int,
    channel_nb: int,
    nb_epoch_fade_in_new_layer: int,
    TrainingJobStatus: str,
    dataset: str,
    self_attention: bool,
    momment_loss: int,
    batch_size: int,
    encoder_network_factor: float,
    target_len: int,
    use_loss: str,
    pos_enc_dimension: int,
    context_length: int = None,
    missing_values_stretch: int = None,
    cold_start_value: float = None,
    LARS: bool = None,
):
    df_filterd = df[
        (df["num_batches_per_epoch"] == num_batches_per_epoch)
        & (df["channel_nb"] == channel_nb)
        & (df["nb_epoch_fade_in_new_layer"] == nb_epoch_fade_in_new_layer)
        & (df["TrainingJobStatus"] == TrainingJobStatus)
        & (df["dataset"] == dataset)
        & (df["self_attention"].isin([self_attention]))
        & (df["momment_loss"] == momment_loss)
        & (df["batch_size"] == batch_size)
        & (
            df["encoder_network_factor"].isnull()
            if encoder_network_factor == 0
            else df["encoder_network_factor"] == encoder_network_factor
        )
        # &(df["num_epochs"]==3500)
        & (df["target_len"] == target_len)
        & (df["use_loss"] == use_loss)
        & (df["pos_enc_dimension"] == pos_enc_dimension)
        & (
            df["context_length"] == min(context_length, target_len)
            if context_length is not None
            else df["context_length"].isnull()
        )
        & (df["LARS"] == LARS if LARS is not None else df["LARS"].isnull())
        & (
            df["missing_values_stretch"] == missing_values_stretch
            if missing_values_stretch is not None
            else df["missing_values_stretch"].isnull()
        )
        & (
            df["cold_start_value"] == cold_start_value
            if cold_start_value is not None
            else df["cold_start_value"].isnull()
        )
    ]
    assert not df_filterd.empty
    return df_filterd


# def psa_gan_TEST(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:  # PSAGAN as in the workshop
#     params_psa_gan = {
#         "num_batches_per_epoch": 10,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 2,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 64,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#     }
#     filtered_df = filter_df(**params_psa_gan)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return model_name


def psa_gan(
    dataset: str, target_len: int, model: int = 0
) -> str:  # PSAGAN as in the workshop
    params_psa_gan = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
    }
    filtered_df = filter_df(**params_psa_gan)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return model_name


# def psa_gan_ABLATION(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:  # PSAGAN as in the workshop
#     params_psa_gan = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": 256,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": False,
#     }
#     filtered_df = filter_df(**params_psa_gan)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return model_name


def psa_gan_wCtx(
    dataset: str, target_len: int, model: int = 0
) -> str:  # PSAGAN with Context
    params_psa_gan = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 16,
    }
    filtered_df = filter_df(**params_psa_gan)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return model_name


def psa_gan_wo_fadein(
    dataset: str, target_len: int, model: int = 0
) -> str:  # PSAGAN without fadein
    params_psa_gan_wo_fadein = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 2,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
    }
    filtered_df = filter_df(**params_psa_gan_wo_fadein)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return model_name


# def psa_gan_wo_fadein_ABLATION(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:  # PSAGAN as in the workshop
#     params_psa_gan = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 2,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": 256,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": False,
#     }
#     filtered_df = filter_df(**params_psa_gan)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return model_name


def psa_gan_wo_ML(
    dataset: str, target_len: int, model: int = 0
) -> str:  # PSAGAN without fadein
    params_psa_gan_wo_ML = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 0,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
    }
    filtered_df = filter_df(**params_psa_gan_wo_ML)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return model_name


# def psa_gan_wo_ML_ABLATION(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:  # PSAGAN as in the workshop
#     params_psa_gan = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 0,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": 256,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": False,
#     }
#     filtered_df = filter_df(**params_psa_gan)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return model_name


def psa_gan_wo_SA(
    dataset: str, target_len: int, model: int = 0
) -> str:  # PSAGAN without fadein
    params_psa_gan_wo_SA = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": False,
        "momment_loss": 1,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
    }
    filtered_df = filter_df(**params_psa_gan_wo_SA)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return model_name


# def psa_gan_wo_SA_ABLATION(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:  # PSAGAN as in the workshop
#     params_psa_gan = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": False,
#         "momment_loss": 1,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": 256,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": False,
#     }
#     filtered_df = filter_df(**params_psa_gan)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return model_name


def psa_gan_wo_ML_wo_fadein(  # PSAGAN without ML without fadein
    dataset: str, target_len: int, model: int = 0
) -> str:
    params_psa_gan_wo_ML_wo_fadein = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 2,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 0,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
    }
    filtered_df = filter_df(**params_psa_gan_wo_ML_wo_fadein)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**params_psa_gan_wo_ML_wo_fadein)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


# def psa_gan_wo_ML_wo_fadein_ABLATION(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:  # PSAGAN as in the workshop
#     params_psa_gan = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 2,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": 256,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": False,
#     }
#     filtered_df = filter_df(**params_psa_gan)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return model_name


def psa_gan_wo_ML_wo_SA(dataset: str, target_len: int, model: int = 0) -> str:
    # PSAGAN without ML and without SA
    params_psa_gan_wo_ML_wo_SA = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": False,
        "momment_loss": 0,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
    }
    filtered_df = filter_df(**params_psa_gan_wo_ML_wo_SA)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**params_psa_gan_wo_ML_wo_SA)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


# def psa_gan_wo_ML_wo_SA_ABLATION(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:  # PSAGAN as in the workshop
#     params_psa_gan = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": False,
#         "momment_loss": 0,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": 256,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": False,
#     }
#     filtered_df = filter_df(**params_psa_gan)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return model_name


def psa_gan_BS256(dataset: str, target_len: int, model: int = 0) -> str:
    # PSAGAN without ML and without SA
    params_psa_gan_BS256 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 256,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
    }
    filtered_df = filter_df(**params_psa_gan_BS256)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**params_psa_gan_BS256)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS512(dataset: str, target_len: int, model: int = 0) -> str:
    # PSAGAN without ML and without SA
    params_psa_gan_BS512 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 512,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
    }
    filtered_df = filter_df(**params_psa_gan_BS512)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**params_psa_gan_BS512)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_wo_SA_wo_fadein(
    dataset: str, target_len: int, model: int = 0
) -> str:
    # PSAGAN without SA and without fadein
    params_psa_gan_wo_SA_wo_fadein = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 2,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": False,
        "momment_loss": 1,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
    }
    return (
        filter_df(**params_psa_gan_wo_SA_wo_fadein)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_wo_ML_wo_SA_wo_fadein(
    dataset: str, target_len: int, model: int = 0
) -> str:
    # PSAGAN without ML, without SA and without fadein
    params_psa_gan_wo_ML_wo_SA_wo_fadein = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 2,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": False,
        "momment_loss": 0,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
    }
    return (
        filter_df(**params_psa_gan_wo_ML_wo_SA_wo_fadein)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS64(dataset: str, target_len: int, model: int = 0) -> str:
    # PSAGAN BS256 PE16
    psa_gan_BS64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 64,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
    }
    filtered_df = filter_df(**psa_gan_BS64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS64).iloc[model].UploadS3Location.split("/")[-1]
    )


# def psa_gan_LARS_BS512(dataset: str, target_len: int, model: int = 0) -> str:
#     # PSAGAN BS256 PE16
#     psa_gan_LARS_BS512 = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 512,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": True,
#     }
#     filtered_df = filter_df(**psa_gan_LARS_BS512)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return (
#         filter_df(**psa_gan_LARS_BS512)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


# def psa_gan_LARS_BS2048(dataset: str, target_len: int, model: int = 0) -> str:
#     # PSAGAN BS256 PE16
#     psa_gan_LARS_BS2048 = {
#         "num_batches_per_epoch": 50,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 2048,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#         "LARS": True,
#     }
#     filtered_df = filter_df(**psa_gan_LARS_BS2048)
#     model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
#     ds_name = dataset.replace("_", "-", 1)
#     assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
#     return (
#         filter_df(**psa_gan_LARS_BS2048)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


def psa_gan_BS256_wCtx64(dataset: str, target_len: int, model: int = 0) -> str:
    # PSAGAN BS256 PE16
    psa_gan_BS256_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 256,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 64,
        "LARS": False,
    }
    filtered_df = filter_df(**psa_gan_BS256_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS256_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS512_wCtx64(dataset: str, target_len: int, model: int = 0) -> str:
    # PSAGAN BS512 PE16
    psa_gan_BS512_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 512,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 64,
        "LARS": False,
    }
    filtered_df = filter_df(**psa_gan_BS512_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS512_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS512_COLD(
    dataset: str,
    cold_start_value: float,
    target_len: int = 256,
    model: int = 0,
) -> str:
    assert cold_start_value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    psa_gan_BS512_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 512,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
        "missing_values_stretch": 0,
        "cold_start_value": cold_start_value,
    }
    filtered_df = filter_df(**psa_gan_BS512_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS512_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS256_COLD(
    dataset: str,
    cold_start_value: float,
    target_len: int = 256,
    model: int = 0,
) -> str:
    assert cold_start_value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    psa_gan_BS512_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 256,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
        "missing_values_stretch": 0,
        "cold_start_value": cold_start_value,
    }
    filtered_df = filter_df(**psa_gan_BS512_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS512_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS128_COLD(
    dataset: str,
    cold_start_value: float,
    target_len: int = 256,
    model: int = 0,
) -> str:
    assert cold_start_value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    psa_gan_BS512_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
        "missing_values_stretch": 0,
        "cold_start_value": cold_start_value,
    }
    filtered_df = filter_df(**psa_gan_BS512_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS512_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS256_MV(
    dataset: str,
    missing_values_stretch: float,
    target_len: int = 256,
    model: int = 0,
) -> str:
    assert missing_values_stretch in [5, 10, 20, 35, 50, 65, 85, 110]
    psa_gan_BS512_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 256,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
        "missing_values_stretch": missing_values_stretch,
        "cold_start_value": 0,
    }
    filtered_df = filter_df(**psa_gan_BS512_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS512_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def TIMEGAN(dataset: str, target_len: int):
    return f"TIMEGAN_{target_len}_{dataset}"


def EBGAN(dataset: str, target_len: int):
    return f"EBGAN_{target_len}_{dataset}"


def EBGAN_MV(dataset: str, missing_value: int):
    return f"EBGAN__256__MV{missing_value}_{dataset}"


def TIMEGAN_MV(dataset: str, missing_value: int):
    return f"TIMEGAN__256__MV{missing_value}_{dataset}"


def TIMEGAN_CS(dataset: str, cold_start_value: int):
    return f"TIMEGAN__256__MV{cold_start_value}_{dataset}".replace(".", "")


def EBGAN_CS(dataset: str, cold_start_value: int):
    return f"EBGAN__256__MV{cold_start_value}_{dataset}".replace(".", "")


def psa_gan_BS512_MV(
    dataset: str,
    missing_values_stretch: float,
    target_len: int = 256,
    model: int = 0,
) -> str:
    assert missing_values_stretch in [5, 10, 20, 35, 50, 65, 85, 110]
    psa_gan_BS512_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 512,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
        "missing_values_stretch": missing_values_stretch,
        "cold_start_value": 0,
    }
    filtered_df = filter_df(**psa_gan_BS512_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS512_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


def psa_gan_BS128_MV(
    dataset: str,
    missing_values_stretch: float,
    target_len: int = 256,
    model: int = 0,
) -> str:
    assert missing_values_stretch in [5, 10, 20, 35, 50, 65, 85, 110]
    psa_gan_BS512_wCtx64 = {
        "num_batches_per_epoch": 100,
        "channel_nb": 32,
        "nb_epoch_fade_in_new_layer": 500,
        "TrainingJobStatus": "Completed",
        "dataset": dataset,
        "self_attention": True,
        "momment_loss": 1,
        "batch_size": 128,
        "encoder_network_factor": 0.0,
        "target_len": target_len,
        "use_loss": "lsgan",
        "pos_enc_dimension": 0,
        "context_length": 0,
        "LARS": False,
        "missing_values_stretch": missing_values_stretch,
        "cold_start_value": 0,
    }
    filtered_df = filter_df(**psa_gan_BS512_wCtx64)
    model_name = filtered_df.iloc[model].UploadS3Location.split("/")[-1]
    ds_name = dataset.replace("_", "-", 1)
    assert f"PSA-GAN-{ds_name}-len{target_len}" in model_name
    return (
        filter_df(**psa_gan_BS512_wCtx64)
        .iloc[model]
        .UploadS3Location.split("/")[-1]
    )


# def psa_gan_PE16(dataset: str, target_len: int, model: int = 0) -> str:
#     # PSAGAN with PE16
#     params_psa_gan_PE16 = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 16,
#     }
#     return (
#         filter_df(**params_psa_gan_PE16)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


# def psa_gan_PE16_BS256_wo_fadein(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:
#     # PSAGAN BS256 PE16 without fadein
#     params_psa_gan_PE16_BS256_wo_fadein = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 2,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 256,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 16,
#     }
#     return (
#         filter_df(**params_psa_gan_PE16_BS256_wo_fadein)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


# def psa_gan_PE16_BS384(dataset: str, target_len: int, model: int = 0) -> str:
#     # PSAGAN BS384 PE16
#     params_psa_gan_PE16_BS384 = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 500,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 384,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 16,
#     }
#     return (
#         filter_df(**params_psa_gan_PE16_BS384)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


# def psa_gan_PE16_BS384_wo_fadein(
#     dataset: str, target_len: int, model: int = 0
# ) -> str:
#     # PSAGAN BS384 PE16 without fadein
#     params_psa_gan_PE16_BS384_wo_fadein = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 2,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 384,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 16,
#     }
#     return (
#         filter_df(**params_psa_gan_PE16_BS384_wo_fadein)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


# def psa_gan_TEST_wCtx(dataset: str, target_len: int, model: int = 0) -> str:
#     # PSAGAN BS384 PE16 without fadein
#     params_psa_gan_TEST_wCtx = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 20,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 10,
#     }
#     return (
#         filter_df(**params_psa_gan_TEST_wCtx)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


# def psa_gan_TEST_woCtx(dataset: str, target_len: int, model: int = 0) -> str:
#     # PSAGAN BS384 PE16 without fadein
#     params_psa_gan_TEST_woCtx = {
#         "num_batches_per_epoch": 100,
#         "channel_nb": 32,
#         "nb_epoch_fade_in_new_layer": 20,
#         "TrainingJobStatus": "Completed",
#         "dataset": dataset,
#         "self_attention": True,
#         "momment_loss": 1,
#         "batch_size": 128,
#         "encoder_network_factor": 0.0,
#         "target_len": target_len,
#         "use_loss": "lsgan",
#         "pos_enc_dimension": 0,
#         "context_length": 0,
#     }
#     return (
#         filter_df(**params_psa_gan_TEST_woCtx)
#         .iloc[model]
#         .UploadS3Location.split("/")[-1]
#     )


if __name__ == "__main__":
    for dataset in ["electricity", "traffic", "m4_hourly", "solar-energy"]:
        for target_len in [16, 32, 64, 128, 256]:

            # for dataset in ["electricity", "traffic", "m4_hourly", "solar-energy"]:
            #     for target_len in [256]:

            logger.info(f"Dataset: {dataset}, Target Len : {target_len}")
            # psa_gan_TEST(dataset=dataset, target_len=target_len)
            psa_gan(dataset=dataset, target_len=target_len)
            psa_gan_wCtx(dataset=dataset, target_len=target_len)
            psa_gan_wo_fadein(dataset=dataset, target_len=target_len)
            psa_gan_wo_ML(dataset=dataset, target_len=target_len)
            psa_gan_wo_SA(dataset=dataset, target_len=target_len)
            psa_gan_wo_ML_wo_fadein(dataset=dataset, target_len=target_len)
            psa_gan_wo_ML_wo_SA(dataset=dataset, target_len=target_len)
            psa_gan_BS256(dataset=dataset, target_len=target_len)
            psa_gan_BS512(dataset=dataset, target_len=target_len)
            psa_gan_wo_ML_wo_SA_wo_fadein(
                dataset=dataset, target_len=target_len
            )
            psa_gan_wo_SA_wo_fadein(dataset=dataset, target_len=target_len)
            # psa_gan_LARS_BS512(dataset=dataset, target_len=target_len)
            # psa_gan_LARS_BS2048(dataset=dataset, target_len=target_len)
            psa_gan_BS64(dataset=dataset, target_len=target_len)
            psa_gan_BS256_wCtx64(dataset=dataset, target_len=target_len)
            psa_gan_BS512_wCtx64(dataset=dataset, target_len=target_len)

            if target_len == 256:
                for cold_start_value in [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ]:
                    logger.info(f"Cold value : {cold_start_value}")
                    psa_gan_BS512_COLD(
                        dataset=dataset,
                        cold_start_value=cold_start_value,
                        target_len=target_len,
                    )
                    psa_gan_BS256_COLD(
                        dataset=dataset,
                        cold_start_value=cold_start_value,
                        target_len=target_len,
                    )
                    psa_gan_BS128_COLD(
                        dataset=dataset,
                        cold_start_value=cold_start_value,
                        target_len=target_len,
                    )
                for missing_val in [5, 50, 110]:
                    logger.info(f"MIssing value : {missing_val}")
                    psa_gan_BS128_MV(
                        dataset=dataset,
                        missing_values_stretch=missing_val,
                        target_len=target_len,
                    )
                    psa_gan_BS256_MV(
                        dataset=dataset,
                        missing_values_stretch=missing_val,
                        target_len=target_len,
                    )
                    psa_gan_BS512_MV(
                        dataset=dataset,
                        missing_values_stretch=missing_val,
                        target_len=target_len,
                    )

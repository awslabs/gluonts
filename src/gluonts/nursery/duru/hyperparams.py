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
from utils import str2bool
import wandb
import os
import pickle5 as pickle  # on Python 3.6 must use this
import argparse


class Hyperparams(dict):
    """
    Wrapper class for a dictionary required for pickling.
    """

    def __init__(self, input_dict):
        for k, v in input_dict.items():
            self[k] = v

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

    # required for pickling !
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


def args_parser():

    parser = argparse.ArgumentParser(description="HVAE hyperparameters.")

    # Most used configs
    parser.add_argument("--user", type=str, default="user")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        metavar="N",
        help="device chosen, one in {'cpu','cuda'}. If using less than all GPUs is desired, use CUDA_VISIBLE_DEVICES=0 for instance before your python command.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        metavar="N",
        help="mode of wandb run tracking, either no tracking ('disabled') or with tracking ('online')",
    )
    parser.add_argument(
        "--model", type=str, default="vdvae_conv", help="one in {vdvae_conv,vdvae_fc}"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="electricity",
        help="one in {sine, 2gauss, electricity, [all other datasets from gluonts]}",
    )
    parser.add_argument("--seed", type=int, default=1)
    # Logs
    parser.add_argument(
        "--iters_per_val",
        type=int,
        default=10000,
        help="number of iterations per evaluation on validation (not test!) dataset.",
    )
    parser.add_argument(
        "--iters_per_vis",
        type=int,
        default=10000,
        help="every how many iterations (=mini-batches) to do visualizations",
    )
    parser.add_argument(
        "--iters_per_train_metrics",
        type=int,
        default=200,
        help="number of iterations after which to log metrics during training",
    )
    parser.add_argument(
        "--iters_per_model_save",
        type=int,
        default=30000,
        help="number of iterations after which model 'checkpoint' is saved, `-1` means do not save model",
    )
    # Training configs
    parser.add_argument(
        "--train_id",
        type=str,
        default=None,
        help="Wandb Run ID (e.g. '3cmfeilu') to continue training of. If None, training a new model from scratch or testing.",
    )
    parser.add_argument(
        "--train_iter",
        type=int,
        default=None,
        help="If `train_id` is specified: The iteration to continue training from. Must have saved model for this iteration. If None, the last saved model state is used to continue training.",
    )
    # Testing configs
    parser.add_argument(
        "--test_id",
        type=str,
        default=None,
        help="Wandb Run ID (e.g. '3cmfeilu') to evaluate on test set. If None, we train a model instead.",
    )
    parser.add_argument(
        "--test_iter",
        type=int,
        default=None,
        help="If `test_id` not None, `test_iter` specifies the model state to test. If None, the last saved model state is used for testing.",
    )

    # ----------------------------------------------------------------------------------------------

    # VDVAE configs
    parser.add_argument(
        "--enc_spec",
        type=str,
        default="3,d2,3",
        help="Encoder specification. comma-separated specification of 1) ResNet blocks in one of two formats '<int>' where <int> are the number of blocks, or '<int1>r<int2>' where <int1> is the number of blocks each repeated <int2> times; 2) down-operations (pooling) in the format 'd<int>' where <int> is the down-scaling factor of the spatial dimension. Examples: '10,d2,5,d2,3r3",
    )
    parser.add_argument(
        "--enc_context_spec",
        type=str,
        default="3,d2,3",
        help="Encoder specification. comma-separated specification of 1) ResNet blocks in one of two formats '<int>' where <int> are the number of blocks, or '<int1>r<int2>' where <int1> is the number of blocks each repeated <int2> times; 2) down-operations (pooling) in the format 'd<int>' where <int> is the down-scaling factor of the spatial dimension. Examples: '10,d2,5,d2,3r3",
    )
    parser.add_argument(
        "--dec_spec",
        type=str,
        default="3,u2,3",
        help="Decoder specification. comma-separated specification of 1) ResNet blocks in one of two formats '<int>' where <int> are the number of blocks, or '<int1>r<int2>' where <int1> is the number of blocks each repeated <int2> times; 2) up-operations (TODO) in the format 'u<int>' where <int> is the down-scaling factor of the spatial dimension. Examples: '10,u2,5,u2,3r3",
    )

    # 1) VDVAEConv
    ## Encoder
    ### BlockConvWithBottleneck (Encoder)
    parser.add_argument(
        "--vdvae_enc_state_channels",
        type=int,
        default=4,
        help="number of channels in the residual state of the encoder.",
    )
    # context window has separate encoder
    parser.add_argument(
        "--vdvae_enc_context_state_channels",
        type=int,
        default=4,
        help="number of channels in the residual state of the encoder of the context window.",
    )
    parser.add_argument(
        "--vdvae_enc_bottleneck_channels_factor",
        type=float,
        default=0.25,
        help="number of channels in bottleneck, as a factor of number of channels in residual state, in the decoder.",
    )
    parser.add_argument(
        "--vdvae_enc_n_conv_3x3",
        type=int,
        default=2,
        help="how many 3x3 conv layers in the ResNet blocks of the encoder to use in the channel bottleneck.",
    )
    ## Decoder
    ### BlockConvWithBottleneck (Encoder)
    parser.add_argument(
        "--vdvae_dec_state_channels",
        type=int,
        default=4,
        help="number of channels in the residual state of the decoder.",
    )
    parser.add_argument(
        "--vdvae_dec_bottleneck_channels_factor",
        type=float,
        default=0.25,
        help="number of channels in bottleneck, as a factor of number of channels in residual state, in the decoder.",
    )
    parser.add_argument(
        "--vdvae_dec_n_conv_3x3",
        type=int,
        default=2,
        help="how many 3x3 conv layers in the ResNet blocks of the decoder to use in the channel bottleneck.",
    )
    # -
    parser.add_argument(
        "--z_channels",
        type=int,
        default=16,
        help="Number of channels of latent variable (spatial resolution defined by model architecture).",
    )

    # 2) VDVAEfc
    ## Encoder
    parser.add_argument(
        "--enc_forecast_state_dim_input",
        type=int,
        default=32,
        help="Initial dimension of the residual state of the encoder. Should be power of two.",
    )
    parser.add_argument(
        "--enc_context_state_dim_input",
        type=int,
        default=32,
        help="Initial dimension of the residual state of the encoder of the context window. Should be power of two.",
    )
    parser.add_argument(
        "--enc_bottleneck_dim_factor",
        type=float,
        default=0.25,
        help="Initial dimension in bottleneck, as a factor of dimension in residual state, in the decoder.",
    )
    parser.add_argument(
        "--enc_n_fc_bottleneck",
        type=int,
        default=2,
        help="How many fc layers in the ResNet blocks of the encoder to use in the channel bottleneck.",
    )
    ## Decoder
    parser.add_argument(
        "--dec_state_dim_input",
        type=int,
        default=32,
        help="Initial dimension of the residual state of the decoder. Should be power of two.",
    )
    parser.add_argument(
        "--dec_bottleneck_dim_factor",
        type=float,
        default=0.25,
        help="Dimension in bottleneck, as a factor of dimension in residual state, in the decoder.",
    )
    parser.add_argument(
        "--dec_n_fc_bottleneck",
        type=int,
        default=2,
        help="How many fc layers in the ResNet blocks of the decoder to use in the bottleneck.",
    )
    # -
    parser.add_argument(
        "--z_dim_factor",
        type=float,
        default=0.5,
        help="Factor of latent variable relative to current resolution.",
    )

    ## Likelihood
    parser.add_argument(
        "--likelihood_type",
        type=str,
        default="GaussianSigmaEstimated",
        help="Which likelihood model to choose. One in {GaussianSigmaHyperparam, GaussianSigmaEstimated}.",
    )
    parser.add_argument(
        "--sigma_p_x_z",
        type=float,
        default=0.01,
        help="If --likelihood_type==GaussianSigmaHyperparam: The STD of the independent Gaussians in p(x|z).",
    )

    # ----------------------------------------------------------------------------------------------

    # Dataset configs
    # TODO also check that they are multiples of 2
    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="Length (=number of timesteps) for the context window. 0 is training an unconditional HVAE.",
    )
    parser.add_argument(
        "--forecast_length",
        type=int,
        default=32,
        help="Length (=number of timesteps) for the forecast window. If --context_length=0, the time series that is unconditionally modelled.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="per_ts_standardize",
        help="choose from {'per_ts_standardize', 'train_data_standardize'}.",
    )
    parser.add_argument(
        "--single_item",
        type=str2bool,
        nargs="?",
        dest="single_item",
        const=True,
        default=False,
        help="whether to just use a single item (single time series) for training and validation on gluonts datasets or not.",
    )
    parser.add_argument(
        "--chosen_id",
        type=int,
        default=1,
        help="If --single_item True, which id that is chosen.",
    )

    # ----------------------------------------------------------------------------------------------

    # Optimizer configs
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100000,
        help="After how many epochs to stop training (typically a very high number and terminate through user).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of items in one mini-batch during training",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--warmup_iters",
        type=float,
        default=100,
        help="Number of iterations during which learning rate is linearly increased in the beginning of training",
    )
    parser.add_argument(
        "--grad_clip_threshold",
        type=float,
        default=200.0,
        help="Threshold of L_2 norm of flattened gradient vector, beyond which the gradient norm is 'clipped', i.e. the gradient vector is scaled s.t. the new gradient norm is equal to the `grad_clip_threshold`\
                                                                                  value of `-1` means the threshold is infinity.",
    )
    parser.add_argument(
        "--grad_skip_threshold",
        type=float,
        default=400.0,
        help="Threshold of L_2 norm of flattened gradient vector, beyond which the current gradient update is skipped and another mini-batch is sampled.\
                                                                                  value of `-1` means the threshold is infinity.",
    )
    parser.add_argument(
        "--ema_rate",
        type=float,
        default=0.9999,
        help="Rate of exponential decay of parameters in the evaluation model in every optimizer step.",
    )
    ## Adam
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay of Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta_1", type=float, default=0.9, help="beta_1 parameter in Adam"
    )
    parser.add_argument(
        "--adam_beta_2", type=float, default=0.9, help="beta_2 parameter in Adam"
    )
    # Other training/eval configs
    parser.add_argument(
        "--val_windows_per_item",
        type=int,
        default=10,
        help="Number of full windows (of length context_length+forecast_length) extracted at the end of training items, used for validation purposes.",
    )
    parser.add_argument(
        "--n_train_iters_per_epoch",
        type=int,
        default=10000,
        help="Number of training samples per epoch.",
    )
    # -
    parser.add_argument(
        "--n_val_iters_per_procedure",
        type=int,
        default=5000,
        help="Number of evaluation batches per evaluation procedure.",
    )
    parser.add_argument(
        "--n_test_iters_per_procedure",
        type=int,
        default=5000,
        help="Number of evaluation batches per evaluation procedure.",
    )

    # ------------------------------------

    # Log configs (details)
    # # -
    parser.add_argument(
        "--vis_train",
        type=str2bool,
        nargs="?",
        dest="vis_train",
        const=True,
        default=True,
        help="whether to do visualizations using the training data or not",
    )
    parser.add_argument(
        "--vis_eval",
        type=str2bool,
        nargs="?",
        dest="vis_eval",
        const=True,
        default=True,
        help="whether to do visualizations using the validation/test data or not",
    )
    # ---
    # Reconstruction
    parser.add_argument(
        "--vis_train_recon",
        type=str2bool,
        nargs="?",
        dest="vis_train_recon",
        const=True,
        default=True,
        help="training data: whether to do visualizations of inputs and reconstructions",
    )
    parser.add_argument(
        "--vis_eval_recon",
        type=str2bool,
        nargs="?",
        dest="vis_eval_recon",
        const=True,
        default=True,
        help="validation/test data: whether to do visualizations of inputs and reconstructions",
    )
    parser.add_argument(
        "--recon_n_rows",
        type=int,
        default=3,
        help="How many rows to fill with reconstructions in the grid plot.",
    )
    parser.add_argument(
        "--recon_n_cols",
        type=int,
        default=3,
        help="How many pairs of inputs and reconstructions to visualize in every row of the grid plot.",
    )
    # ---
    # Unconditional samples
    parser.add_argument(
        "--vis_p_sample",
        type=str2bool,
        nargs="?",
        dest="vis_p_sample",
        const=True,
        default=True,
        help="Whether to do visualizations p samples.",
    )
    parser.add_argument(
        "--p_sample_n_ts_per_temp",
        type=int,
        default=3,
        help="How many unconditional samples to visualize per temperature (row) the grid plot.",
    )
    parser.add_argument(
        "--p_sample_temp_list",
        type=list,
        default=[1.0, 1.0, 1.0],
        metavar="N",
        help="The temperatures corresponding to rows in the grid plot filled with unconditional samples.",
    )  # [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    parser.add_argument(
        "--p_sample_n_samples",
        type=int,
        default=50,
        help="Number of samples (from p(z)) for each 'prediction' we want to make.",
    )
    # ---
    # Posterior-prior comparison
    # number of samples is controlled by `recon_n_rows` and `recon_n_pairs_col`
    parser.add_argument(
        "--vis_eval_posterior_prior",
        type=str2bool,
        nargs="?",
        dest="vis_eval_posterior_prior",
        const=True,
        default=True,
        help="Whether to do visualizations of the posterior-prior comparison, both 'per resolution' and from 'first to last'.",
    )
    parser.add_argument(
        "--posterior_prior_temp",
        type=float,
        default=0.9,
        help="Temperature for sampling in the posterior prior plot.",
    )
    # ---
    # Cumulative KL
    parser.add_argument(
        "--vis_train_cum_kl",
        type=str2bool,
        nargs="?",
        dest="vis_train_cum_kl",
        const=True,
        default=True,
        help="training data: whether to visualize the cumulative KLs of one plot or not",
    )
    parser.add_argument(
        "--vis_eval_cum_kl",
        type=str2bool,
        nargs="?",
        dest="vis_eval_cum_kl",
        const=True,
        default=True,
        help="training data: whether to visualize the cumulative KLs of one plot or not",
    )
    parser.add_argument(
        "--cum_kl_n_batches",
        type=int,
        default=10,
        help="How many batches to use to produce the KL cum sum plots.",
    )
    # ---
    # State norm plots
    parser.add_argument(
        "--vis_eval_state_norm_enc",
        type=str2bool,
        nargs="?",
        dest="vis_eval_state_norm_enc",
        const=True,
        default=True,
        help="Whether to visualise state norm plot in encoder.",
    )
    parser.add_argument(
        "--vis_eval_state_norm_enc_context",
        type=str2bool,
        nargs="?",
        dest="vis_eval_state_norm_enc_context",
        const=True,
        default=True,
        help="Whether to visualise state norm plot in encoder.",
    )
    parser.add_argument(
        "--vis_eval_state_norm_dec",
        type=str2bool,
        nargs="?",
        dest="vis_eval_state_norm_dec",
        const=True,
        default=True,
        help="Whether to visualise state norm plot in encoder.",
    )
    parser.add_argument(
        "--state_norm_n_batches",
        type=int,
        default=10,
        help="How many batches to use to produce the state norm plots.",
    )

    # -------------------------------------

    # initialised during runtime (through code)
    # TODO this shouldn't be command line arguments, rather they should be added to H later on
    parser.add_argument(
        "--n_meas",
        type=int,
        default=None,
        help="Number of measurements (=covariates) at one time step. Equal to number of 'channels'.",
    )
    parser.add_argument(
        "--restore_iter",
        type=int,
        default=None,
        help="The iteration of which a model state was restored from for continued training or testing purposes.",
    )
    parser.add_argument(
        "--n_params", type=str, default=None, help="Number of model parameters."
    )
    parser.add_argument(
        "--conditional",
        type=str2bool,
        nargs="?",
        dest="conditional",
        const=True,
        default=None,
        help="whether to have an unconditional (single window) or conditional (context and forecast window) task. Determined by context_length",
    )
    parser.add_argument(
        "--pad_context",
        type=int,
        default=None,
        help="Padding (just one side) to be done for context window.",
    )
    parser.add_argument(
        "--pad_forecast",
        type=int,
        default=None,
        help="Padding (just one side) to be done for forecast window.",
    )

    args = parser.parse_args()
    H = Hyperparams(args.__dict__)

    return H


def check_hyperparams(H):

    if H.train_id is not None and H.test_id is not None:
        raise Exception(
            "Restoring training from a specific run and testing a specific run at the same time is not possible."
        )

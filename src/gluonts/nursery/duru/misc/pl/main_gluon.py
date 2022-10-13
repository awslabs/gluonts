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
import wandb
import torch
import os
import numpy as np

from hyperparams import args_parser
from utils import (
    load_dict_from_yaml,
    check_hyperparams,
    download_some_wandb_files,
)
from data import set_up_data

# from plotting import plot_inputs_and_recons_torch_grid, plot_uncond_samples  # TODO import
# from analysis import posterior_prior_front_to_back, posterior_prior_per_res  # TODO import

from misc.pl.pl import VDVAEConvPL

from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
)

# -
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify

# -

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def main():
    H = args_parser()
    # check hyperparams for consistency etc.
    check_hyperparams(H)

    # make entire code deterministic
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)

    # load wandb api key, project name and entity name
    wandb_config = load_dict_from_yaml("../../setup/wandb.yml")
    # login to wandb and create run
    wandb.login(key=wandb_config[H.user])
    wandb_run = wandb.init(
        project=wandb_config["project_name"],
        entity=wandb_config["team_name"],
        mode=H.wandb_mode,
    )

    # additionally, create a WandbLogger class for PyTorch Lightning
    wandb_logger = WandbLogger(
        project=wandb_config["project_name"],
        entity=wandb_config["team_name"],
        mode=H.wandb_mode,
    )

    # auxiliary / remembering variables
    restore = H.train_id is not None or H.test_id is not None
    training = H.test_id is None
    train_iter = H.train_iter
    test_iter = H.test_iter

    # download some files first and load them
    if restore:
        files_to_restore = [
            "H.dict",
            "last_save_iter.th",
        ]  # 'last_save_iter.th' is just storing an int
        run_id = H.train_id if H.train_id is not None else H.test_id
        download_some_wandb_files(
            files_to_restore=files_to_restore, run_id=run_id
        )
        # Note: loads another H dictionary in the case of restoring which overwrites the new one above
        H = torch.load(
            os.path.join(wandb.run.dir, files_to_restore[0])
        )  # overwrites H parsed above
        last_save_iter = torch.load(
            os.path.join(wandb.run.dir, files_to_restore[1])
        )
        # In the restored H, we overwrite train or test restore information which we need below
        if training:
            H.train_id = run_id
            H.train_iter = train_iter
        else:
            H.test_id = run_id
            H.test_iter = test_iter
        print(
            "Note: Restoring run "
            + run_id
            + ". Any passed command line arguments are ignored!"
        )  # Note: Could even throw exception if this is the case.

    # updating the hyperparameters:
    # for sweep: don't use such values of H above which are defined by sweep
    # set args value to sweep values instead
    # only do this when training (as during testing, the config has been set to the config of the run)
    if training:
        for (key, value) in wandb.config.items():
            setattr(H, key, value)  # args is namespace object

    if H.train_id is not None:
        train_iter = last_save_iter if H.train_iter is None else H.train_iter
        H.restore_iter = train_iter
        model_load_file_name = "iter-%d-model.th" % train_iter
        model_eval_load_file_name = "iter-%d-model_eval.th" % train_iter
        # optimizer_load_file_name = 'iter-%d-optimizer.th'%train_iter
        # scheduler_load_file_name = 'iter-%d-scheduler.th'%train_iter
        files_to_restore = [
            model_load_file_name,
            model_eval_load_file_name,
        ]  # , optimizer_load_file_name, scheduler_load_file_name
        download_run_id = H.train_id
    elif H.test_id is not None:
        test_iter = last_save_iter if H.test_iter is None else H.test_iter
        H.restore_iter = test_iter
        # Note: could only load model_eval here
        model_load_file_name = "iter-%d-model.th" % test_iter
        model_eval_load_file_name = "iter-%d-model_eval.th" % test_iter
        # optimizer_load_file_name = 'iter-%d-optimizer.th'%test_iter
        # scheduler_load_file_name = 'iter-%d-scheduler.th'%test_iter
        files_to_restore = [
            model_load_file_name,
            model_eval_load_file_name,
        ]  # , optimizer_load_file_name, scheduler_load_file_name
        download_run_id = H.test_id
    else:
        train_iter = 0

    if restore:
        download_some_wandb_files(
            files_to_restore=files_to_restore, run_id=download_run_id
        )

    # make device a global variable so that dataset.py can access it
    global device
    # initializing global variable (see above)
    device = torch.device(H.device)

    # set up data
    dataset_train, dataset_val, dataset_test, preprocess_fn = set_up_data(H)

    # add how many batches per epoch on each dataset split
    # TODO are the below correctly used?
    H.n_train_iters_per_epoch = int(H.n_train_samples / H.batch_size)
    # TODO should it be that validation and test set always have the same number of samples?
    H.n_val_iters_per_epoch = int(H.n_eval_samples / H.batch_size)
    H.n_test_iters_per_epoch = int(H.n_eval_samples / H.batch_size)

    # save the hyperparam config after everything was set up completely (some values of H will only be written at run-time, but everything is written at this point)
    if not H.test_eval and not H.train_restore:
        print("saving H config...")
        torch.save(H, os.path.join(wandb.run.dir, "H.dict"))

    # initialise models from scratch (even if restoring)
    # Note: during testing, could only load test model, but not done for now
    if H.model == "vdvae_conv":
        hvae_eval = VDVAEConvPL(H=H)
        hvae = VDVAEConvPL(H=H, hvae_eval=hvae_eval)

    if restore:
        # this also loads the optimizer and learning rate scheduler, since the state dicts are those of pl modules, see https://forums.pytorchlightning.ai/t/saving-and-loading-optimizer-state/1533
        hvae.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, model_load_file_name))
        )
        hvae_eval.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, model_eval_load_file_name))
        )

    # correctly parallelise the models
    if torch.cuda.is_available() and "cuda" in H.device:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        print("Using CPU!")

    # Not needed because handled in PL
    # if torch.cuda.device_count() > 1 and 'cuda' in H.device:
    #     # parallelising both hvae and hvae_eval (note: consider parallelise just hvae)
    #     # documentation: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    #     # TODO implement DistributedDataParallel for further speed up
    #     hvae = torch.nn.DataParallel(hvae)
    #     # TODO HVAE_EVAL not done with DataParallel, simply for the reason every call of function of the module would have to be called via
    #     # TODO hvae_eval.module.function rather than hvae_eval.function
    #     # hvae_eval = torch.nn.aaParallel(hvae_eval)

    # handled by PL
    # if 'cuda' in H.device:
    #     # transfer model to GPU
    #     # device = torch.device(H.device)  #  'cuda' or 'cpu'
    #     hvae = hvae.cuda()  #  .to(device=device)
    #     hvae_eval = hvae_eval.cuda()  # .to(device=device)

    # weights&biases tracking (gradients, network topology)
    # only watch vae (this is the one doing the training step)
    # only do so when training
    if H.test_id is None:
        wandb.watch(hvae)

    # count total number of parameters and print
    params_count = sum(
        p.numel() for p in hvae.parameters() if p.requires_grad
    )  #  ; 7754180
    params_string = f"{params_count:,}"
    print("Number of parameters of HVAE: " + params_string)
    # store number of parameters count in configs
    H.n_params = params_string

    # print H and run dir once everything is set up and completed
    print(H)
    print("wandb.run.dir: ", wandb.run.dir)

    # update configs -> remember hyperparams; only when not testing
    if H.test_id is None:
        wandb.config.update(H)

    # TODO attach as instance variable to model --------------
    # load up fixed samples from training set (used for reconstructions only)
    # vis_train_x, vis_train_x_input = get_sample_for_visualization(dataset_train, preprocess_fn=preprocess_fn, n_images=H.recon_n_rows * H.recon_n_pairs_col, device=H.device, conditional=H.conditional)
    # if H.conditional:
    #     x_orig_context, x_orig_forecast = vis_train_x
    #     x_input_context, x_input_forecast = vis_train_x_input
    # if 'cuda' in H.device:
    #     if H.conditional:
    #         x_orig_context, x_orig_forecast = x_orig_context.cuda(non_blocking=True), x_orig_forecast.cuda(non_blocking=True)
    #         x_input_context, x_input_forecast = x_input_context.cuda(non_blocking=True), x_input_forecast.cuda(non_blocking=True)
    #     else:
    #         vis_train_x, vis_train_x_input = vis_train_x.cuda(non_blocking=True), vis_train_x_input.cuda(non_blocking=True)
    # vis_train_x_list = [vis_train_x[i].cpu().detach() for i in range(vis_train_x.shape[0])]

    # load up fixed samples from test set (used for reconstructions only)
    # vis_eval_x, vis_eval_x_input = get_sample_for_visualization(dataset=dataset_eval, preprocess_fn=preprocess_fn,
    #                                                             n_images=self.H.recon_n_rows * self.H.recon_n_pairs_col,
    #                                                             device=self.H.device, conditional=self.H.conditional)
    # if self.H.conditional:
    #     x_orig_context, x_orig_forecast = vis_eval_x
    #     x_input_context, x_input_forecast = vis_eval_x_input
    # if 'cuda' in self.H.device:
    #     if self.H.conditional:
    #         x_orig_context, x_orig_forecast = x_orig_context.cuda(non_blocking=True), x_orig_forecast.cuda(
    #             non_blocking=True)
    #         x_input_context, x_input_forecast = x_input_context.cuda(non_blocking=True), x_input_forecast.cuda(
    #             non_blocking=True)
    #     else:
    #         vis_train_x, vis_train_x_input = vis_eval_x.cuda(non_blocking=True), vis_eval_x_input.cuda(
    #             non_blocking=True)
    #
    # # TODO change after adaptaptation to context and forecast
    # vis_eval_x_list = [vis_eval_x[j].cpu().detach() for j in range(vis_eval_x.shape[0])]
    # TODO ---------------------------

    if H.test_id is not None:
        print("main(): do testing...")
        # delete unused datasets
        del dataset_train
        del dataset_val
        # do testing
        with torch.no_grad():
            test(
                H=H,
                dataset_eval=dataset_test,
                preprocess_fn=preprocess_fn,
                hvae_eval=hvae_eval,
                wandb_logger=wandb_logger,
                i=test_iter,
            )
    else:
        # delete unused datasets
        del dataset_test
        # do training
        print("main(): do training...")
        train(
            H=H,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            preprocess_fn=preprocess_fn,
            hvae=hvae,
            hvae_eval=hvae_eval,
            wandb_logger=wandb_logger,
            i=train_iter,
        )


def train(
    H,
    dataset_train,
    dataset_val,
    preprocess_fn,
    hvae,
    hvae_eval,
    wandb_logger,
    i,
):
    # initialised before training
    hvae.early_iterations = set([1] + [2**exp for exp in range(2, 13)])
    (
        hvae.elbo_has_nan_count,
        hvae.distortion_has_nan_count,
        hvae.rate_has_nan_count,
        hvae.grad_clip_count,
        hvae.skipped_updates_count,
        hvae.grad_skip_count,
        hvae.nan_skip_count,
    ) = (0, 0, 0, 0, 0, 0, 0)
    hvae.i = i

    # TODO understand what each of these do
    mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=H.forecast_length,
        ),
        past_length=H.context_length,
        future_length=H.forecast_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )

    # set up train data loader. take only subset because otherwise too many samples
    subset_indices = np.random.choice(
        np.arange(dataset_train.__len__()),
        size=H.n_train_samples,
        replace=False,
    )
    loader_train = TrainDataLoader(
        Cached(torch.utils.data.Subset(dataset_train, indices=subset_indices)),
        batch_size=H.batch_size,
        stack_fn=batchify,
        transform=mask_unobserved + training_splitter,
        num_batches_per_epoch=H.n_train_iters_per_epoch,
    )

    # set up validation data loader. take only subset because otherwise too many samples
    # subset_indices = np.random.choice(np.arange(dataset_val.__len__()), size=H.n_eval_samples, replace=False)
    # loader_val = ValidationDataLoader(Cached(torch.utils.data.Subset(dataset_val, indices=subset_indices)), batch_size=H.batch_size)

    # TODO potentially pass further arguments
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=H.n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit(
        hvae, train_dataloaders=loader_train
    )  # TODO take validation back in: val_dataloaders=loader_val, val_check_interval=H.iters_per_val


def test(H, dataset_test, preprocess_fn, hvae_eval, wandb_logger, i):
    # set up test data loader. take only subset because otherwise too many samples
    subset_indices = np.random.choice(
        np.arange(dataset_test.__len__()), size=H.n_eval_samples, replace=False
    )
    # TODO still using ValidationDataLoader even though used for test set correct?
    loader_test = ValidationDataLoader(
        torch.utils.data.Subset(Cached(dataset_test), indices=subset_indices),
        batch_size=H.batch_size,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    trainer.test(hvae_eval, dataloaders=loader_test)


if __name__ == "__main__":
    print(torch.__version__)
    main()

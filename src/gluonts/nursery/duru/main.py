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
import time
import matplotlib.pyplot as plt
import pandas

from hyperparams import args_parser
from utils import (
    load_dict_from_yaml,
    check_hyperparams,
    download_some_wandb_files,
    compute_gradient_norm,
    get_sample_for_visualization,
    append_to_list_,
    finites_only,
    compute_blocks_per_res,
)
from data import set_up_data
from duru.hvae import HVAE_dummy
from duru.vdvae_conv import VDVAEConv
from duru.optim import init_optimizer_scheduler, optimizer_step_hvae_eval
from plotting import (
    plot_kl_cum_sum,
    plot_state_norm,
    plot_inputs_and_recons,
    plot_p_sample,
)

# from plotting import plot_inputs_and_recons_torch_grid  # TODO import
# from analysis import posterior_prior_front_to_back, posterior_prior_per_res  # TODO import
from utils import insert_channel_dim
from duru.vdvae_fc import VDVAEfc

from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
)
from gluonts.transform.sampler import PredictionSplitSampler

# -
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify

# -
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation import make_evaluation_predictions, Evaluator

# from torch.utils.tensorboard import SummaryWriter

# anomaly detection to e.g. spot where NaNs/infs are coming from. Makes running slow.
# torch.autograd.set_detect_anomaly(True)


def main():
    H = args_parser()
    # check hyperparams for consistency etc.
    check_hyperparams(H)

    # make entire code deterministic
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)

    # load wandb api key, project name and entity name
    wandb_config = load_dict_from_yaml("setup/wandb.yml")
    # login to wandb and create run
    wandb.login(key=wandb_config[H.user])
    wandb_run = wandb.init(
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

    # whether conditional or not (for convenience)
    if H.context_length != 0:
        H.conditional = True
    else:
        H.conditional = False

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
        optimizer_load_file_name = "iter-%d-optimizer.th" % train_iter
        scheduler_load_file_name = "iter-%d-scheduler.th" % train_iter
        files_to_restore = [
            model_load_file_name,
            model_eval_load_file_name,
            optimizer_load_file_name,
            scheduler_load_file_name,
        ]
        download_run_id = H.train_id
    elif H.test_id is not None:
        test_iter = last_save_iter if H.test_iter is None else H.test_iter
        H.restore_iter = test_iter
        # Note: could only load model_eval here
        model_load_file_name = "iter-%d-model.th" % test_iter
        model_eval_load_file_name = "iter-%d-model_eval.th" % test_iter
        optimizer_load_file_name = "iter-%d-optimizer.th" % test_iter
        scheduler_load_file_name = "iter-%d-scheduler.th" % test_iter
        files_to_restore = [
            model_load_file_name,
            model_eval_load_file_name,
            optimizer_load_file_name,
            scheduler_load_file_name,
        ]
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
    (
        dataset_train,
        dataset_val,
        dataset_test,
        normalize_fn,
        unnormalize_fn,
    ) = set_up_data(H)

    # save the hyperparam config after everything was set up completely (some values of H will only be written at run-time, but everything is written at this point)
    if not H.test_eval and not H.train_restore:
        print("saving H config...")
        torch.save(H, os.path.join(wandb.run.dir, "H.dict"))

    # initialise models from scratch (even if restoring)
    # Note: during testing, could only load test model, but not done for now
    if H.model == "dummy":
        hvae = HVAE_dummy(H=H)
        hvae_eval = HVAE_dummy(H=H)
    elif H.model == "vdvae_conv":
        hvae = VDVAEConv(H=H)
        hvae_eval = VDVAEConv(H=H)
    elif H.model == "vdvae_fc":
        hvae = VDVAEfc(H=H)
        hvae_eval = VDVAEfc(H=H)

    # for forward function (evaluator): add normalize and unnormalize function as method to objects
    hvae.normalize_fn = normalize_fn
    hvae_eval.normalize_fn = normalize_fn
    hvae.unnormalize_fn = unnormalize_fn
    hvae_eval.unnormalize_fn = unnormalize_fn

    # initialize from scratch (even if restoring) --> might later load state dict
    optimizer, scheduler = init_optimizer_scheduler(H, hvae)

    if restore:
        hvae.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, model_load_file_name))
        )
        hvae_eval.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, model_eval_load_file_name))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, optimizer_load_file_name))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, scheduler_load_file_name))
        )

    # correctly parallelise the models
    if torch.cuda.is_available() and "cuda" in H.device:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        print("Using CPU!")

    if torch.cuda.device_count() > 1 and "cuda" in H.device:
        # parallelising both hvae and hvae_eval (note: consider parallelise just hvae)
        # documentation: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # TODO implement DistributedDataParallel for further speed up
        hvae = torch.nn.DataParallel(hvae)
        # TODO HVAE_EVAL not done with DataParallel, simply for the reason every call of function of the module would have to be called via
        # TODO hvae_eval.module.function rather than hvae_eval.function
        # hvae_eval = torch.nn.aaParallel(hvae_eval)
    if "cuda" in H.device:
        # transfer model to GPU
        # device = torch.device(H.device)  #  'cuda' or 'cpu'
        hvae = hvae.cuda()  #  .to(device=device)
        hvae_eval = hvae_eval.cuda()  # .to(device=device)

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

    if H.test_id is not None:
        print("main(): do testing...")
        # delete unused datasets
        del dataset_train
        del dataset_val
        # do testing
        with torch.no_grad():
            evaluate(
                H=H,
                dataset_eval=dataset_test,
                normalize_fn=normalize_fn,
                unnormalize_fn=unnormalize_fn,
                hvae_eval=hvae_eval,
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
            normalize_fn=normalize_fn,
            unnormalize_fn=unnormalize_fn,
            hvae=hvae,
            hvae_eval=hvae_eval,
            optimizer=optimizer,
            scheduler=scheduler,
            i=train_iter,
        )


def train(
    H,
    dataset_train,
    dataset_val,
    normalize_fn,
    unnormalize_fn,
    hvae,
    hvae_eval,
    optimizer,
    scheduler,
    i,
):
    # print("begin train()...")
    # setting up masking and instance (=chunk) transformers
    train_mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    n_ts_train = len(dataset_train)
    train_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=(H.n_train_iters_per_epoch * H.batch_size)
            / n_ts_train,  # TODO this value must be set correctly
            min_future=H.forecast_length,
            min_past=H.context_length,
        ),
        past_length=H.context_length,
        future_length=H.forecast_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )

    # load up fixed samples from training set (used for reconstructions only)
    dummy_loader_train = TrainDataLoader(
        Cached(dataset_train),
        batch_size=H.recon_n_rows * H.recon_n_cols,
        stack_fn=batchify,
        transform=train_mask_unobserved
        + train_splitter,  # TODO potentially simply add normalize_fn here
        num_batches_per_epoch=H.n_train_iters_per_epoch,
    )
    dummy_gen_train = iter(dummy_loader_train)

    (
        vis_train_x_context_orig,
        vis_train_x_forecast_orig,
        vis_train_x_context_input,
        vis_train_x_forecast_input,
        vis_train_x_item_id,
    ) = get_sample_for_visualization(
        generator=dummy_gen_train, normalize_fn=normalize_fn, device=H.device
    )
    vis_train_x_context_orig_list = [
        vis_train_x_context_orig[i].cpu().detach()
        for i in range(vis_train_x_context_orig.shape[0])
    ]
    vis_train_x_forecast_orig_list = [
        vis_train_x_forecast_orig[i].cpu().detach()
        for i in range(vis_train_x_forecast_orig.shape[0])
    ]

    # initialised before training
    early_iterations = set([1] + [2**exp for exp in range(2, 13)])
    (
        elbo_has_nan_count,
        distortion_has_nan_count,
        rate_has_nan_count,
        grad_clip_count,
        skipped_updates_count,
        grad_skip_count,
        nan_skip_count,
    ) = (0, 0, 0, 0, 0, 0, 0)

    for epoch in range(H.n_epochs):
        # set up train data loader. take only subset because otherwise too many samples
        if H.dataset == "dummy":
            subset_indices = np.random.choice(
                np.arange(dataset_train.__len__()),
                size=H.n_train_samples,
                replace=False,
            )
            loader_train = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset_train, indices=subset_indices),
                batch_size=H.batch_size,
                drop_last=True,
                pin_memory=True,
                shuffle=True,
            )
        elif H.dataset in list(dataset_recipes.keys()) + ["sine", "2gauss"]:
            loader_train = TrainDataLoader(
                Cached(dataset_train),
                batch_size=H.batch_size,
                stack_fn=batchify,
                transform=train_mask_unobserved
                + train_splitter,  # TODO potentially simply add normalize_fn here
                num_batches_per_epoch=H.n_train_iters_per_epoch,
            )
            gen_train = iter(loader_train)

        for _ in range(loader_train.length):
            # print("train", i)
            t0_train = time.time()

            batch = next(gen_train)
            x_context, x_forecast = (
                batch["past_target"],
                batch["future_target"],
            )
            x_item_id = batch["item_id"]
            # TODO can make nicer: wrap into transform

            if "cuda" in H.device:
                x_context = x_context.cuda(non_blocking=True)
                x_forecast = x_forecast.cuda(non_blocking=True)

            # insert measurement dimension if not existent
            x_context = insert_channel_dim(x_context)
            x_forecast = insert_channel_dim(x_forecast)

            x_context = normalize_fn(x=x_context, ids=x_item_id)
            x_forecast = normalize_fn(x=x_forecast, ids=x_item_id)

            # x_input, x_target = x_input.cuda(non_blocking=True), x_target.cuda(non_blocking=True)
            hvae.zero_grad()
            # TODO actually, negative ELBO, distortion and rate (in nats or bits per dim) are returned
            # TODO filter out isfinite from elbo etc. ? jus like done in evaluate? --> see accumulate_stats function in old code
            # try implemented due to issue in Normal(...) calls, as loc argument sometimes is passed nan values (see https://wandb.ai/diffmodels/hvae/runs/2gmxiz61/logs?wbreferrer=run-alert&workspace=user-user)

            # adding tensorboard (just for looking at the graph)
            # writer = SummaryWriter('tensorboard/test1')

            try:
                if H.conditional:
                    # writer.add_graph(hvae, (x_forecast, x_context))
                    # writer.close()
                    (
                        elbo,
                        distortion,
                        rate,
                        kl_list,
                        state_norm_enc_list,
                        state_norm_enc_context_list,
                        state_norm_dec_list,
                        p_x_z,
                    ) = hvae.forward_regular(
                        x_context=x_context, x_forecast=x_forecast
                    )
                else:
                    (
                        elbo,
                        distortion,
                        rate,
                        kl_list,
                        state_norm_enc_list,
                        state_norm_dec_list,
                        p_x_z,
                    ) = hvae.forward_regular(x_forecast=x_forecast)

                if "cuda" in H.device and torch.cuda.device_count() > 1:
                    # more than one GPU are used with DataParallel --> chunks the batch into torch.cuda.device_count() parts, which are processed on a replicated model on each of the GPUs.
                    # Warning from documentation: "When module returns a scalar (i.e., 0-dimensional tensor) in forward(), this wrapper will return a vector of length equal to number of devices used in data parallelism, containing the result from each device."
                    # this is why a tensor is now suddenly returned for every value which has been a scalar before
                    # we account for that by averaging these tensors (of length torch.cuda.device_count())
                    elbo, distortion, rate = (
                        torch.mean(elbo),
                        torch.mean(distortion),
                        torch.mean(rate),
                    )

                elbo.backward()

                # compute other metrics
                p_x_z_mean = p_x_z.mean
                mse = torch.nn.functional.mse_loss(
                    p_x_z_mean, x_forecast[:, :, H.pad_forecast :]
                )  # slice out the padding of x_forecast, already done in likelihood for p_x_z_mean

                # TODO taken out for now due to bug --> take back in
                # if H.grad_clip_threshold != -1:
                #     grad_norm_before_clipping = torch.nn.utils.clip_grad_norm_(hvae.parameters(), max_norm=H.grad_clip_threshold, norm_type=2.0, error_if_nonfinite=True).item()
                #     grad_clip_count += 1
                # else:
                #     # just compute the gradient norm
                grad_norm_before_clipping = compute_gradient_norm(
                    hvae.parameters()
                )

            except ValueError as e:
                print(e)  # print out exception as if it was thrown
                skipped_updates_count += 1
                nan_skip_count += 1
                # Note: this implementation may cause very high values for counted logger metrics,
                # if exception is thrown and logging in this iteration should be done (because logging and resetting of counters is skipped)
                continue

            elbo_nan_count = torch.isnan(elbo).sum()
            distortion_nan_count = torch.isnan(distortion).sum()
            rate_nan_count = torch.isnan(rate).sum()
            elbo_has_nan_count += 1 if elbo_nan_count > 0 else 0
            distortion_has_nan_count += 1 if distortion_nan_count > 0 else 0
            rate_has_nan_count += 1 if rate_nan_count > 0 else 0

            # perform optimizer step (updating the parameters) if 1) gradient norm before clipping is below `H.grad_skip_threshold` 2) neither the distortion nor rate contains NANs
            if (
                (
                    H.grad_skip_threshold == -1
                    or grad_norm_before_clipping < H.grad_skip_threshold
                )
                and distortion_nan_count == 0
                and rate_nan_count == 0
            ):
                optimizer.step()
                optimizer_step_hvae_eval(hvae, hvae_eval, H.ema_rate)
            else:
                skipped_updates_count += 1
                grad_skip_count += 1
            # regardless of whether the optimizer took a step, update the scheduler
            scheduler.step()

            t1_train = time.time()
            time_in_sec_training_step = t1_train - t0_train  #  time in seconds

            # train metrics logging
            if i % H.iters_per_train_metrics == 0 or i in early_iterations:
                # Note: kl_list not logged
                log_dict = {
                    "train/epoch": i / H.n_train_iters_per_epoch,
                    "train/elbo": elbo,
                    "train/distortion": distortion,
                    "train/rate": rate,
                    "train/mse": mse,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/elbo_has_nan_count": elbo_has_nan_count,
                    "train/distortion_has_nan_count": distortion_has_nan_count,
                    "train/rate_has_nan_count": rate_has_nan_count,
                    "train/train_step_seconds": time_in_sec_training_step,
                    "train/grad_norm_before_clipping": grad_norm_before_clipping,
                    "train/skipped_updates": skipped_updates_count,
                    "train/nan_skip_count": nan_skip_count,
                    "train/grad_skip_count": grad_skip_count,
                }
                wandb.log(log_dict, step=i)
                # reset counters
                (
                    elbo_has_nan_count,
                    distortion_has_nan_count,
                    rate_has_nan_count,
                    grad_clip_count,
                    skipped_updates_count,
                    nan_skip_count,
                    grad_skip_count,
                ) = (0, 0, 0, 0, 0, 0, 0)

            # training data visualizations
            with torch.no_grad():
                if H.vis_train and i % H.iters_per_vis == 0:

                    if H.vis_train_recon:
                        x_context = (
                            vis_train_x_context_input
                            if H.conditional
                            else None
                        )
                        if (
                            torch.cuda.device_count() > 1
                            and "cuda" in H.device
                        ):
                            recon_x_hat, recon_p_x_z = hvae.module.get_recon(
                                x_forecast=vis_train_x_forecast_input,
                                x_context=x_context,
                            )
                        else:
                            recon_x_hat, recon_p_x_z = hvae.get_recon(
                                x_forecast=vis_train_x_forecast_input,
                                x_context=x_context,
                            )
                        recon_x_hat = unnormalize_fn(
                            recon_x_hat, ids=vis_train_x_item_id
                        )  # unnormalize
                        # unnormalize parameters of p_x_z distribution objects --> create a new one
                        # TODO not nice: recreating the Normal object, becasue likelihood/model dependent
                        if H.likelihood_type == "GaussianSigmaHyperparam":
                            recon_p_x_z = hvae.likelihood_model.get_p_x_z(
                                p_x_z_mean=unnormalize_fn(
                                    recon_p_x_z.mean, ids=vis_train_x_item_id
                                )
                            )
                        elif H.likelihood_type == "GaussianSigmaEstimated":
                            recon_p_x_z = hvae.likelihood_model.get_p_x_z(
                                p_x_z_mean=unnormalize_fn(
                                    recon_p_x_z.mean, ids=vis_train_x_item_id
                                ),
                                p_x_z_log_std=unnormalize_fn(
                                    torch.log(recon_p_x_z.stddev),
                                    ids=vis_train_x_item_id,
                                ),
                            )

                        recon_x_hat_list = [
                            recon_x_hat[i].cpu().detach()
                            for i in range(recon_x_hat.shape[0])
                        ]  # convert to list

                        # TODO this is the line where p_x_z implementation no longer works --> continue implementing
                        # recon_p_x_z_list = [recon_p_x_z[i].cpu().detach() for i in range(recon_x_hat.shape[0])]
                        recon_p_x_z_list = None

                        if H.conditional:
                            recon_x_context_list = [
                                vis_train_x_context_orig_list[i].cpu().detach()
                                for i in range(recon_x_hat.shape[0])
                            ]
                        else:
                            recon_x_context_list = None
                        # x_list = [vis_train_x_forecast_input[i].cpu().detach() for i in range(vis_train_x_forecast_input.shape[0])]
                        fig = plot_inputs_and_recons(
                            x_list=vis_train_x_forecast_orig_list,
                            x_hat_list=recon_x_hat_list,
                            x_context_list=recon_x_context_list,
                            recon_n_rows=H.recon_n_rows,
                            recon_n_cols=H.recon_n_cols,
                        )
                        wandb.log(
                            {
                                "train_vis"
                                + "/inputs_and_reconstructions": wandb.Image(
                                    plt
                                )
                            },
                            step=i,
                        )
                        plt.close(fig=fig)  # close the figure
                    if H.vis_train_cum_kl:
                        kl_cum_sum = torch.cumsum(
                            input=torch.stack(kl_list, dim=1), dim=1
                        ).cpu()  # this uses the KLs of one mini-batch, could instead accumulate a couple of batches; cpu (and later numpy) conversion required for plotting
                        fig = plot_kl_cum_sum(
                            kl_cum_sum=kl_cum_sum, dataset=H.dataset
                        )
                        log_key = (
                            "train_vis" + "/Cumulative, batch-averaged KLs"
                        )
                        wandb.log({log_key: wandb.Image(plt)}, step=i)
                        plt.close(fig=fig)  # close the figure

            # Evaluate model on the validation (not test!) set
            if i % H.iters_per_val == 0:
                with torch.no_grad():
                    evaluate(
                        H=H,
                        dataset_eval=dataset_val,
                        normalize_fn=normalize_fn,
                        unnormalize_fn=unnormalize_fn,
                        hvae_eval=hvae_eval,
                        i=i,
                    )

            # Models, optimizer and scheduler saving 'checkpoint'
            if (
                i % H.iters_per_model_save == 0
                and not H.iters_per_model_save == -1
            ):  # -1 means not saving model
                prefix = "iter-%d-" % (i)
                torch.save(
                    hvae.state_dict(),
                    os.path.join(wandb.run.dir, prefix + "model.th"),
                )
                torch.save(
                    hvae_eval.state_dict(),
                    os.path.join(wandb.run.dir, prefix + "model_eval.th"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(wandb.run.dir, prefix + "optimizer.th"),
                )
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(wandb.run.dir, prefix + "scheduler.th"),
                )
                torch.save(
                    i, os.path.join(wandb.run.dir, "last_save_iter.th")
                )  # save without prefix

            # in very end of iteraiton: increment iteration count
            i += 1

            # print("train i: %d"%(i))


def evaluate(H, dataset_eval, normalize_fn, unnormalize_fn, hvae_eval, i):
    t0_eval = time.time()

    eval_mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )

    if H.test_id is None:

        n_ts_eval = len(dataset_eval)
        if H.test_id is not None:
            n_batches = H.n_test_iters_per_procedure
        else:
            n_batches = H.n_val_iters_per_procedure
        num_instances = (n_batches * H.batch_size) / n_ts_eval
        eval_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            # PredictionSplitSampler
            instance_sampler=PredictionSplitSampler(
                min_past=H.context_length, min_future=H.forecast_length
            ),
            # instance_sampler=ExpectedNumInstanceSampler(  #
            #     num_instances=num_instances,  # TODO is this value correct?
            #     min_future=H.forecast_length,
            # ),
            past_length=H.context_length,
            future_length=H.forecast_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )

        # set up validation data loader. take only subset because otherwise too many samples
        if H.dataset == "dummy":
            subset_indices = np.random.choice(
                np.arange(dataset_eval.__len__()),
                size=H.n_eval_samples,
                replace=False,
            )
            loader_eval = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset_eval, indices=subset_indices),
                batch_size=H.batch_size,
                drop_last=True,
                pin_memory=True,
                shuffle=True,
            )
        elif H.dataset in list(dataset_recipes.keys()) + ["sine", "2gauss"]:
            # An iterable sequence of batches, not a data loader.
            # This is what's also stated in the Returns value in the documentation, yet, very misleading because inconsistent with TrainDataLoader.
            # See https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.loader.html?highlight=validationdataloader#gluonts.dataset.loader.ValidationDataLoader.
            loader_eval = ValidationDataLoader(
                dataset_eval,  # Cached(dataset_eval)
                batch_size=H.batch_size,
                stack_fn=batchify,
                transform=eval_mask_unobserved + eval_splitter,
                # TODO potentially simply add normalize_fn here
            )
            gen_eval = iter(loader_eval)

        (
            elbo_list,
            elbo_filtered_list,
            distortion_list,
            rate_list,
            kl_list_list,
            state_norm_enc_list_list,
            state_norm_enc_context_list_list,
            state_norm_dec_list_list,
            mse_list,
            mae_list,
            rmse_list,
        ) = ([], [], [], [], [], [], [], [], [], [], [])

        count_eval_iters = 0

        while True:  # loop is terminated via try except below
            # print("eval loop", count_eval_iters)
            # TODO 3/9 do try except also in train?
            # print("while - before")
            try:
                batch = next(gen_eval)
            except StopIteration:
                break
            # print("while - after")

            x_context, x_forecast = (
                batch["past_target"],
                batch["future_target"],
            )
            x_item_id = batch["item_id"]
            # TODO can make nicer: wrap into transform

            if "cuda" in H.device:
                x_context = x_context.cuda(non_blocking=True)
                x_forecast = x_forecast.cuda(non_blocking=True)

            # insert measurement dimension if not existent
            x_context = insert_channel_dim(x_context)
            x_forecast = insert_channel_dim(x_forecast)

            x_context = normalize_fn(x=x_context, ids=x_item_id)
            x_forecast = normalize_fn(x=x_forecast, ids=x_item_id)

            if H.conditional:
                (
                    elbo,
                    distortion,
                    rate,
                    kl_list,
                    state_norm_enc_list,
                    state_norm_enc_context_list,
                    state_norm_dec_list,
                    p_x_z,
                ) = hvae_eval.forward_regular(
                    x_context=x_context, x_forecast=x_forecast
                )
            else:
                (
                    elbo,
                    distortion,
                    rate,
                    kl_list,
                    state_norm_enc_list,
                    state_norm_dec_list,
                    p_x_z,
                ) = hvae_eval.forward_regular(x_forecast=x_forecast)
                state_norm_enc_context_list = [None]  # dummy

            if "cuda" in H.device and torch.cuda.device_count() > 1:
                # more than one GPU are used with DataParallel --> chunks the batch into torch.cuda.device_count() parts, which are processed on a replicated model on each of the GPUs.
                # Warning from documentation: "When module returns a scalar (i.e., 0-dimensional tensor) in forward(), this wrapper will return a vector of length equal to number of devices used in data parallelism, containing the result from each device."
                # this is why a tensor is now suddenly returned for every value which has been a scalar before
                # we account for that by averaging these tensors (of length torch.cuda.device_count())
                elbo, distortion, rate = (
                    torch.mean(elbo),
                    torch.mean(distortion),
                    torch.mean(rate),
                )

            # compute other metrics
            # note that these metrics are computed on the unnormalised time series
            pred = hvae_eval.sample_p(
                n_samples=H.p_sample_n_samples,
                temp=1.0,
                x_context=x_context,
                set_z_sample=None,
            )
            # reshape so to get batch dimension back
            # do not use reshape!!!
            pred = pred.unsqueeze(0)
            pred = torch.split(
                pred, split_size_or_sections=H.p_sample_n_samples, dim=1
            )
            pred = torch.cat(pred, dim=0)
            # take median as the prediction
            pred = torch.quantile(pred, dim=1, q=0.5, keepdim=False)
            # pred = pred.flatten(start_dim=0, end_dim=1)  # sample dimension flattened  # TODO not done, because unnormalize_fn needs measurement dimension (i.e. sample dimension is now measurement dimension
            # pred = p_x_z.mean

            # unnormalize
            x_context = unnormalize_fn(x_context, ids=x_item_id)
            x_forecast = unnormalize_fn(x_forecast, ids=x_item_id)
            pred = unnormalize_fn(pred, ids=x_item_id)

            mse = torch.nn.functional.mse_loss(
                pred, x_forecast[:, :, H.pad_forecast :]
            )  # slice out the padding of x_forecast, already done in likelihood for p_x_z_mean
            mae = torch.nn.functional.l1_loss(
                pred, x_forecast[:, :, H.pad_forecast :]
            )
            rmse = torch.sqrt(mse)

            # Note: might have to average across machines when using DataParallel --> see main.py, ll. 98
            # Note: if this blows up memory: could only append a certain number of KLs and state norms
            append_to_list_(
                (elbo, elbo_list),
                (distortion, distortion_list),
                (rate, rate_list),
                (kl_list, kl_list_list),
                (state_norm_enc_list, state_norm_enc_list_list),
                (
                    state_norm_enc_context_list,
                    state_norm_enc_context_list_list,
                ),
                (state_norm_dec_list, state_norm_dec_list_list),
                (mse, mse_list),
                (mae, mae_list),
                (rmse, rmse_list),
            )

            count_eval_iters += 1
            # print("eval j: %d"%(j))

        # compute filtered metrics for elbo, distortion, rate (might contain infinite or nan values)
        (
            elbo_list,
            distortion_list,
            rate_list,
            mse_list,
            mae_list,
            rmse_list,
        ) = (
            [elbo.cpu().numpy() for elbo in elbo_list],
            [distortion.cpu().numpy() for distortion in distortion_list],
            [rate.cpu().numpy() for rate in rate_list],
            [mse.cpu().numpy() for mse in mse_list],
            [mae.cpu().numpy() for mae in mae_list],
            [rmse.cpu().numpy() for rmse in rmse_list],
        )  # convert ot numpy
        (
            elbo_filtered_list,
            distortion_filtered_list,
            rate_filtered_list,
        ) = finites_only(elbo_list, distortion_list, rate_list)
        # compute means
        (
            elbo_mean,
            distortion_mean,
            rate_mean,
            mse_mean,
            mae_mean,
            rmse_mean,
            elbo_filtered_mean,
            distortion_filtered_mean,
            rate_filtered_mean,
        ) = (
            np.mean(elbo_list),
            np.mean(distortion_list),
            np.mean(rate_list),
            np.mean(mse_list),
            np.mean(mae_list),
            np.mean(rmse_list),
            np.mean(elbo_filtered_list),
            np.mean(distortion_filtered_list),
            np.mean(rate_filtered_list),
        )

        # load up fixed samples from test set (used for reconstructions only)
        dummy_loader_eval = ValidationDataLoader(
            dataset_eval,  # Cached(dataset_eval)
            batch_size=H.batch_size,
            stack_fn=batchify,
            transform=eval_mask_unobserved + eval_splitter,
            # TODO potentially simply add normalize_fn here
            # num_batches_per_epoch=H.n_eval_iters_per_epoch
        )
        dummy_gen_eval = iter(dummy_loader_eval)

        # TODO does passing dummy_gen_eval  work for sine wave dataset? --> also in train
        (
            vis_eval_x_context_orig,
            vis_eval_x_forecast_orig,
            vis_eval_x_context_input,
            vis_eval_x_forecast_input,
            vis_eval_x_item_id,
        ) = get_sample_for_visualization(
            generator=dummy_gen_eval,
            normalize_fn=normalize_fn,
            device=H.device,
        )
        vis_eval_x_context_orig_list = [
            vis_eval_x_context_orig[i].cpu().detach()
            for i in range(vis_eval_x_context_orig.shape[0])
        ]
        vis_eval_x_forecast_orig_list = [
            vis_eval_x_forecast_orig[i].cpu().detach()
            for i in range(vis_eval_x_forecast_orig.shape[0])
        ]

        with torch.no_grad():
            if H.vis_eval and i % H.iters_per_vis == 0:

                if H.vis_eval_recon:
                    x_context = (
                        vis_eval_x_context_input if H.conditional else None
                    )
                    if torch.cuda.device_count() > 1 and "cuda" in H.device:
                        recon_x_hat, recon_p_x_z = hvae_eval.module.get_recon(
                            x_forecast=vis_eval_x_forecast_input,
                            x_context=x_context,
                        )
                    else:
                        recon_x_hat, recon_p_x_z = hvae_eval.get_recon(
                            x_forecast=vis_eval_x_forecast_input,
                            x_context=x_context,
                        )
                    recon_x_hat = unnormalize_fn(
                        recon_x_hat, ids=vis_eval_x_item_id
                    )  # unnormalize
                    recon_x_hat_list = [
                        recon_x_hat[i].cpu().detach()
                        for i in range(recon_x_hat.shape[0])
                    ]  # convert to list
                    if H.conditional:
                        recon_x_context_list = [
                            vis_eval_x_context_orig_list[i].cpu().detach()
                            for i in range(recon_x_hat.shape[0])
                        ]
                    else:
                        recon_x_context_list = None
                    fig = plot_inputs_and_recons(
                        x_list=vis_eval_x_forecast_orig_list,
                        x_hat_list=recon_x_hat_list,
                        x_context_list=recon_x_context_list,
                        recon_n_rows=H.recon_n_rows,
                        recon_n_cols=H.recon_n_cols,
                    )
                    eval_type = (
                        "test_vis" if H.test_id is not None else "val_vis"
                    )
                    wandb.log(
                        {
                            eval_type
                            + "/inputs_and_reconstructions": wandb.Image(plt)
                        },
                        step=i,
                    )
                    plt.close(fig=fig)  # close the figure

                if H.vis_p_sample:
                    ts_list = []
                    if H.conditional:
                        x_context = vis_eval_x_context_input
                        x_context_list = [
                            vis_eval_x_context_orig_list[i].cpu().detach()
                            for i in range(x_context.shape[0])
                        ]
                        # repeat each item id n_samples times
                        x_item_id = []
                        for id in vis_eval_x_item_id:
                            for _ in range(H.p_sample_n_samples):
                                x_item_id.append(id)
                    else:
                        x_context = None
                        x_item_id = vis_eval_x_item_id

                    for temp in H.p_sample_temp_list:
                        p_samples = hvae_eval.sample_p(
                            n_samples=H.p_sample_n_samples,
                            x_context=x_context,
                            temp=temp,
                        )
                        p_samples = unnormalize_fn(
                            p_samples, ids=x_item_id
                        )  # unnormalize

                        # reshape so to get batch dimension back
                        # do not do reshape!!!
                        p_samples = p_samples.unsqueeze(
                            0
                        )  # new batch dimension
                        p_samples = torch.split(
                            p_samples,
                            split_size_or_sections=H.p_sample_n_samples,
                            dim=1,
                        )
                        p_samples = torch.cat(p_samples, dim=0)
                        # p_samples = p_samples.reshape((len(vis_eval_x_item_id), H.p_sample_n_samples, H.n_meas, H.forecast_length))  # reshape into (batch_size, sample_shape, event_dim_1)

                        ts_list += [
                            p_samples[i].cpu().detach()
                            for i in range(p_samples.shape[0])
                        ]
                        # Note: this implies that every row of uncond sample plot is conditioned on the same training samples,
                        # if H.conditional:
                        #     context_list += [x_context[i].cpu().detach() for i in range(vis_eval_x_context_input.shape[0])]
                    if H.conditional:
                        # in spite of sampling from p, we use here te reconstructions plot in order to visualise also the ground truth prediction
                        fig = plot_inputs_and_recons(
                            x_list=vis_eval_x_forecast_orig_list,
                            x_hat_list=ts_list,
                            do_samples=True,
                            x_context_list=x_context_list,
                            recon_n_rows=len(H.p_sample_temp_list),
                            recon_n_cols=H.p_sample_n_ts_per_temp,
                        )
                        # fig = plot_p_sample(ts_list=ts_list, x_context_list=context_list, p_sample_n_rows=len(H.p_sample_temp_list), p_sample_n_cols=vis_eval_x_context_input.shape[0])  # reason: the get_sample_vis batch is used
                    else:
                        fig = plot_p_sample(
                            ts_list=ts_list,
                            x_context_list=x_context_list,
                            p_sample_n_rows=len(H.p_sample_temp_list),
                            p_sample_n_cols=H.p_sample_n_ts_per_temp,
                        )  # TODO
                    if H.test_id is not None:
                        type = "test_vis"
                    else:
                        type = "val_vis"
                    wandb.log({type + "/p_samples": wandb.Image(plt)}, step=i)
                    plt.close(fig=fig)  # close the figure

                # TODO
                # if H.vis_eval_posterior_prior:
                #     # 'front to back'
                #     fig = posterior_prior_front_to_back(H, hvae_eval, vis_eval_x, vis_eval_x_input)
                #     eval_type = 'test_vis' if H.test_id is not None else 'val_vis'
                #     wandb.log({eval_type + "/posterior-prior comparison": wandb.Image(plt)}, step=i)
                #     plt.close(fig=fig)  # close the figure
                #     # 'per resolution'
                #     fig = posterior_prior_per_res(H, hvae_eval, vis_eval_x, vis_eval_x_input)
                #     eval_type = 'test_vis' if H.test_id is not None else 'val_vis'
                #     wandb.log({eval_type + "/posterior-prior-per-res comparison": wandb.Image(plt)}, step=i)
                #     plt.close(fig=fig)  # close the figure

                if H.vis_eval_cum_kl:
                    # Note: when using the below code, and calling torch.cumsum with kl_list_concat instead of kl_list, one could use all eval batches instead of just one.
                    #               However, this would make the KL plot inconsistent between train and eval, and likewise, if the batch size is large enough, should not make a big difference anyway.
                    #               Hence not done for now.
                    # kl_list_list = kl_list_list[:H.cum_kl_n_batches]
                    # kl_list_concat = []
                    # for i, kl_list in enumerate(kl_list_list):
                    #     for j, kl in enumerate(kl_list):
                    #         if i == 0:
                    #             kl_list_concat.append(kl)
                    #         else:
                    #             kl_list_concat[j] = torch.cat((kl_list_concat[j], kl))

                    kl_cum_sum = torch.cumsum(
                        input=torch.stack(kl_list, dim=1), dim=1
                    ).cpu()  # this uses the KLs of one mini-batch, could instead accumulate a couple of batches; cpu (and later numpy) conversion required for plotting
                    fig = plot_kl_cum_sum(
                        kl_cum_sum=kl_cum_sum, dataset=H.dataset
                    )
                    eval_type = (
                        "test_vis" if H.test_id is not None else "val_vis"
                    )
                    log_key = eval_type + "/Cumulative, batch-averaged KLs"
                    wandb.log({log_key: wandb.Image(plt)}, step=i)
                    plt.close(fig=fig)  # close the figure

                if H.vis_eval_state_norm_enc:
                    # TODO consider adding input of 0th block (s.t. starts at 0)
                    # TODO consider implementing the same plot for training
                    state_norm_enc_list_list = state_norm_enc_list_list[
                        : H.state_norm_n_batches
                    ]
                    state_norm_enc = [
                        torch.stack(state_norm_list, dim=1)
                        for state_norm_list in state_norm_enc_list_list
                    ]
                    state_norm_enc = torch.cat(state_norm_enc, dim=0)
                    state_norm_enc = state_norm_enc.detach().cpu().numpy()

                    res_to_n_layers = compute_blocks_per_res(
                        spec=H.enc_spec,
                        enc_or_dec="enc",
                        input_resolution=H.forecast_length,
                        count_up_down=True,
                    )

                    fig = plot_state_norm(
                        state_norm=state_norm_enc,
                        enc_or_dec="enc",
                        res_to_n_layers=res_to_n_layers,
                    )
                    eval_type = (
                        "test_vis" if H.test_id is not None else "val_vis"
                    )
                    log_key = eval_type + "/State L2-norm (Encoder)"
                    wandb.log({log_key: wandb.Image(plt)}, step=i)
                    plt.close(fig=fig)  # close the figure

                if H.vis_eval_state_norm_enc_context and H.conditional:
                    # TODO consider adding input of 0th block (s.t. starts at 0)
                    # TODO consider implementing the same plot for training
                    state_norm_enc_context_list_list = (
                        state_norm_enc_context_list_list[
                            : H.state_norm_n_batches
                        ]
                    )
                    state_norm_enc_context = [
                        torch.stack(state_norm_list, dim=1)
                        for state_norm_list in state_norm_enc_context_list_list
                    ]
                    state_norm_enc_context = torch.cat(
                        state_norm_enc_context, dim=0
                    )
                    state_norm_enc_context = (
                        state_norm_enc_context.detach().cpu().numpy()
                    )

                    res_to_n_layers = compute_blocks_per_res(
                        spec=H.enc_context_spec,
                        enc_or_dec="enc",
                        input_resolution=H.context_length,
                        count_up_down=True,
                    )

                    fig = plot_state_norm(
                        state_norm=state_norm_enc_context,
                        enc_or_dec="enc",
                        res_to_n_layers=res_to_n_layers,
                    )
                    eval_type = (
                        "test_vis" if H.test_id is not None else "val_vis"
                    )
                    log_key = eval_type + "/State L2-norm (Encoder context)"
                    wandb.log({log_key: wandb.Image(plt)}, step=i)
                    plt.close(fig=fig)  # close the figure

                if H.vis_eval_state_norm_dec:
                    # TODO consider adding input of 0th block (s.t. starts at 0)
                    # TODO consider implementing the same plot for training
                    state_norm_dec_list_list = state_norm_dec_list_list[
                        : H.state_norm_n_batches
                    ]
                    state_norm_dec = [
                        torch.stack(state_norm_list, dim=1)
                        for state_norm_list in state_norm_dec_list_list
                    ]
                    state_norm_dec = torch.cat(state_norm_dec, dim=0)
                    state_norm_dec = state_norm_dec.detach().cpu().numpy()

                    res_to_n_layers = compute_blocks_per_res(
                        spec=H.dec_spec,
                        enc_or_dec="dec",
                        input_resolution=H.forecast_length,
                        count_up_down=True,
                    )

                    fig = plot_state_norm(
                        state_norm=state_norm_dec,
                        enc_or_dec="dec",
                        res_to_n_layers=res_to_n_layers,
                    )
                    eval_type = (
                        "test_vis" if H.test_id is not None else "val_vis"
                    )
                    log_key = eval_type + "/State L2-norm (Decoder)"
                    wandb.log({log_key: wandb.Image(plt)}, step=i)
                    plt.close(fig=fig)  # close the figure

            t1_eval = time.time()
            time_in_sec_training_step = t1_eval - t0_eval  #  time in seconds

            # do logging of scalar metrics
            # Note: kl_list, n_batches_eval not logged
            if H.test_id is not None:
                eval_type = "test/"
            else:
                eval_type = "val/"  #
            log_dict = {
                # eval_type + 'epoch': i / H.n_train_iters_per_epoch,
                eval_type + "elbo": elbo_mean,
                eval_type + "distortion": distortion_mean,
                eval_type + "rate": rate_mean,
                eval_type + "mse": mse_mean,
                eval_type + "mae": mae_mean,
                eval_type + "rmse": rmse_mean,
                eval_type + "elbo_filtered": elbo_filtered_mean,
                eval_type + "distortion_filtered": distortion_filtered_mean,
                eval_type + "rate_filtered": rate_filtered_mean,
                eval_type
                + "total_eval_time_seconds": time_in_sec_training_step,
            }

            wandb.log(
                log_dict, step=i
            )  # "As long as you keep passing the same value for step, W&B will collect the keys and values from each call in one unified dictionary. As soon you call wandb.log() with a different value for step than the previous one, W&B will write all the collected keys and values to the history"  (see https://docs.wandb.ai/guides/track/log)

            print("Evaluating. Iter %d: " % (i), log_dict)

    else:
        prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=H.context_length,
            future_length=H.forecast_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )

        predictor_hvae_eval = hvae_eval.get_predictor(
            eval_mask_unobserved + prediction_splitter
        )

        # set seed once again s.t. the same samples are drawn every single time
        np.random.seed(H.seed)
        torch.manual_seed(H.seed)
        torch.cuda.manual_seed(H.seed)

        x_hat_it, ts_it = make_evaluation_predictions(
            dataset=dataset_eval,
            predictor=predictor_hvae_eval,
            num_samples=H.p_sample_n_samples,
        )

        x_hat_list, ts_list = list(x_hat_it), list(ts_it)

        evaluator = Evaluator(quantiles=[0.05, 0.5, 0.95])
        metrics_dict, per_ts_metrics = evaluator(ts_list, x_hat_list)
        # metrics_df = pandas.DataFrame.from_records(metrics_torch, index=["FeedForward"]).transpose()

        log_dict = {}
        for k, v in metrics_dict.items():
            log_dict["test/" + k] = v
        # TODO for some reason, the dictionary is not actually properly logged
        wandb.log(
            log_dict, step=i
        )  # "As long as you keep passing the same value for step, W&B will collect the keys and values from each call in one unified dictionary. As soon you call wandb.log() with a different value for step than the previous one, W&B will write all the collected keys and values to the history"  (see https://docs.wandb.ai/guides/track/log)

        print("Testing. Iter %d: " % (i), log_dict)


if __name__ == "__main__":
    print(torch.__version__)
    main()

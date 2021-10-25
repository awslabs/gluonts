import os
import json
import torch
import torch.nn as nn
import argparse
import matplotlib
import numpy as np
from tensorboardX import SummaryWriter
import src.utils as utils
import src.datasets as datasets
import src.tensorboard_utils as tensorboard_utils
from src.model_utils import build_model
from src.evaluation import evaluate_gts_dataset


def train_step(batch, model, optimizer, step, config, device):
    model.train()

    def _set_lr(lr):
        for g in optimizer.param_groups:
            g['lr'] = lr
    switch_temp = utils.get_temperature(step, config, 'switch_')
    extra_args = dict()
    dur_temp = 1.
    if config['model'] == 'REDSDS':
        dur_temp = utils.get_temperature(step, config, 'dur_')
        extra_args = {'dur_temperature': dur_temp}
    lr = utils.get_learning_rate(step, config)
    xent_coeff = utils.get_cross_entropy_coef(step, config)
    cont_ent_anneal = config['cont_ent_anneal']
    optimizer.zero_grad()
    result = model(
        batch['past_target'].to(device),
        ctrl_inputs=dict(
            feat_static_cat=batch['feat_static_cat'].to(device),
            past_time_feat=batch['past_time_feat'].to(device)),
        switch_temperature=switch_temp,
        cont_ent_anneal=cont_ent_anneal,
        num_samples=config['num_samples'],
        **extra_args)
    objective = -1 * (result[config['objective']] +
                      xent_coeff * result['crossent_regularizer'])

    print(
        step,
        f'obj: {objective.item():.4f}',
        f'lr: {lr:.6f}',
        f's-temp: {switch_temp:.2f}',
        f'cross-ent: {xent_coeff}',
        f'cont ent: {cont_ent_anneal}',
    )

    objective.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
    _set_lr(lr)
    optimizer.step()
    result['objective'] = objective
    result['lr'] = lr
    result['switch_temperature'] = switch_temp
    result['dur_temperature'] = dur_temp
    result['xent_coeff'] = xent_coeff
    return result


def plot_results(result, prefix=''):
    original_inputs = result["inputs"][0].data.cpu().numpy()
    reconstructed_inputs = result["reconstructions"][0].data.cpu().numpy()
    most_likely_states = torch.argmax(
        result["log_gamma"],
        dim=-1)[0][0].data.cpu().numpy()
    hidden_states = result["x_samples"][0].data.cpu().numpy()
    discrete_states_lk = torch.exp(
        result["log_gamma"][0])[0].data.cpu().numpy()
    true_seg = None
    if 'true_seg' in result.keys():
        true_seg = result['true_seg'][
            0, :config['context_length']].data.cpu().numpy()

    matplotlib_fig = tensorboard_utils.show_time_series(
        fig_size=(12, 4),
        inputs=original_inputs,
        reconstructed_inputs=reconstructed_inputs,
        segmentation=most_likely_states,
        true_segmentation=true_seg,
        fig_title="input_reconstruction")
    fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
    summary.add_image(
        f'{prefix}Reconstruction', fig_numpy_array, step, dataformats='HWC')

    matplotlib_fig = tensorboard_utils.show_hidden_states(
        fig_size=(12, 3),
        zt=hidden_states,
        segmentation=most_likely_states)
    fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
    summary.add_image(
        f'{prefix}Hidden_State_xt', fig_numpy_array, step, dataformats='HWC')

    matplotlib_fig = tensorboard_utils.show_discrete_states(
        fig_size=(12, 3),
        discrete_states_lk=discrete_states_lk,
        segmentation=most_likely_states)
    fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
    summary.add_image(
        f'{prefix}Discrete_State_zt', fig_numpy_array, step, dataformats='HWC')


if __name__ == "__main__":
    matplotlib.use('Agg')

    # COMMAND-LINE ARGS
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config',
        type=str,
        help='Path to config file.')
    group.add_argument(
        '--ckpt',
        type=str,
        help='Path to checkpoint file.')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Which device to use, e.g., cpu, cuda:0, cuda:1, ...')
    args = parser.parse_args()

    # CONFIG
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        config = ckpt['config']
    else:
        config = utils.get_config_and_setup_dirs(args.config)
    device = torch.device(args.device)

    # DATA
    train_dataset = datasets.GTSUnivariateDataset(
        config['dataset'], batch_size=config['batch_size'])
    val_dataset = datasets.GTSUnivariateDataset(
        config['dataset'], batch_size=10, mode='val')
    test_dataset = datasets.GTSUnivariateDataset(
        config['dataset'], batch_size=10, mode='test')

    # NOTE: batch_size is None because we are handling batching outselves
    # in the dataset.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=None, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=None)

    # Replacing config values with dataset metadata values.
    config['context_length'] = test_dataset.metadata['context_length']
    config['prediction_length'] = test_dataset.metadata['prediction_length']
    config['freq'] = test_dataset.metadata['freq']
    config['control']['n_staticfeat'] = test_dataset.metadata['n_staticfeat']
    config['control']['n_timefeat'] = test_dataset.metadata['n_timefeat']

    train_gen = iter(train_loader)
    val_gen = iter(val_loader)

    # MODEL
    model = build_model(config=config)
    start_step = 1
    if args.ckpt:
        model.load_state_dict(ckpt['model'])
        start_step = ckpt['step'] + 1
    model = model.to(device)

    for n, p in model.named_parameters():
        print(n, p.size())

    # TRAIN AND EVALUATE
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=config['weight_decay'])
    if args.ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    summary = SummaryWriter(logdir=config['log_dir'])

    all_metrics = {
        'CRPS': [],
        'MSE': [],
        'step': []
    }

    for step in range(start_step, config['num_steps'] + 1):
        try:
            train_batch = next(train_gen)
        except StopIteration:
            train_gen = iter(train_loader)
        train_result = train_step(
            train_batch, model, optimizer, step, config, device)

        if step % config['save_steps'] == 0 or step == config['num_steps']:
            model_path = os.path.join(config['model_dir'], f'model_{step}.pt')
            torch.save({'step': step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config},
                       model_path)

        if step % config['log_steps'] == 0 or step == config['num_steps']:
            try:
                val_batch = next(val_gen)
            except StopIteration:
                val_gen = iter(val_loader)
            summary_items = {
                "params/learning_rate": train_result['lr'],
                "params/switch_temperature":
                    train_result['switch_temperature'],
                "params/dur_temperature": train_result['dur_temperature'],
                "params/cross_entropy_coef": train_result['xent_coeff'],
                "elbo/training": train_result[config['objective']],
                "xent/training": train_result['crossent_regularizer'],
            }

            plot_results(train_result)

            if step == config['num_steps']:
                # Evaluate Forecast
                agg_metrics = evaluate_gts_dataset(
                    test_dataset,
                    model,
                    device=device,
                    num_samples=config['forecast']['num_samples'],
                    deterministic_z=config['forecast']['deterministic_z'],
                    deterministic_x=config['forecast']['deterministic_x'],
                    deterministic_y=config['forecast']['deterministic_y'],
                    max_len=np.inf,
                    batch_size=100)
                summary_items['metrics/test_mse'] = agg_metrics['MSE']
                summary_items['metrics/CRPS'] =\
                    agg_metrics['mean_wQuantileLoss']
                all_metrics['step'].append(step)
                all_metrics['CRPS'].append(agg_metrics['mean_wQuantileLoss'])
                all_metrics['MSE'].append(agg_metrics['MSE'])

            # Forecast and Plot
            ctx_len = config['context_length']
            past_target = val_batch['past_target'][0:1, :ctx_len]
            feat_static_cat = val_batch['feat_static_cat'][0:1]
            past_time_feat = val_batch['past_time_feat'][0:1, :ctx_len]
            future_time_feat = val_batch['past_time_feat'][0:1, ctx_len:]
            rec_with_forecast = model.predict(
                past_target.to(device),
                ctrl_inputs=dict(
                    feat_static_cat=feat_static_cat.to(device),
                    past_time_feat=past_time_feat.to(device),
                    future_time_feat=future_time_feat.to(device)),
                num_samples=config['forecast']['num_samples'],
                deterministic_z=config['forecast']['deterministic_z'],
                deterministic_x=config['forecast']['deterministic_x'],
                deterministic_y=config['forecast']['deterministic_y'],
                )['rec_n_forecast'].data.cpu().numpy()[:, 0, ...]
            complete_ts = val_batch['past_target'][0:1]
            matplotlib_fig = tensorboard_utils.show_time_series_forecast(
                fig_size=(12, 4),
                inputs=complete_ts.data.cpu().numpy()[0],
                rec_with_forecast=rec_with_forecast,
                context_length=config['context_length'],
                prediction_length=config['prediction_length'],
                fig_title="forecast")
            fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
            summary.add_image(
                'Forecast', fig_numpy_array, step, dataformats='HWC')

            for k, v in summary_items.items():
                summary.add_scalar(k, v, step)
            summary.flush()

with open(os.path.join(config['log_dir'], 'metrics.json'), 'w') as fp:
    json.dump(all_metrics, fp)

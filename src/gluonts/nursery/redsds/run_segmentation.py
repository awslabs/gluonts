import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib
from tensorboardX import SummaryWriter
import src.utils as utils
import src.datasets as datasets
import src.tensorboard_utils as tensorboard_utils
from src.model_utils import build_model
from src.evaluation import evaluate_segmentation
from src.torch_utils import torch2numpy

available_datasets = {
    'bouncing_ball',
    '3modesystem',
    'bee'
}


def train_step(batch, model, optimizer, step, config):
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
        batch, switch_temperature=switch_temp,
        num_samples=config['num_samples'],
        cont_ent_anneal=cont_ent_anneal,
        **extra_args,
    )
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
    original_inputs = torch2numpy(result["inputs"][0])
    reconstructed_inputs = torch2numpy(result["reconstructions"][0])
    most_likely_states = torch2numpy(
        torch.argmax(
            result["log_gamma"],
            dim=-1)[0][0]
    )
    hidden_states = torch2numpy(result["x_samples"][0])
    discrete_states_lk = torch2numpy(
        torch.exp(
            result["log_gamma"][0])[0]
    )
    true_seg = None
    if 'true_seg' in result:
        true_seg = torch2numpy(
            result['true_seg'][0, :config['context_length']]
        )

    ylim = 1.3 * np.abs(original_inputs).max()
    matplotlib_fig = tensorboard_utils.show_time_series(
        fig_size=(12, 4),
        inputs=original_inputs,
        reconstructed_inputs=reconstructed_inputs,
        segmentation=most_likely_states,
        true_segmentation=true_seg,
        fig_title="input_reconstruction",
        ylim=(-ylim, ylim))
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


def get_dataset(dataset):
    assert dataset in available_datasets, (
        f'Unknown dataset {dataset}!'
    )
    if dataset == 'bouncing_ball':
        train_dataset = datasets.BouncingBallDataset(
            path='./data/bouncing_ball.npz'
        )
        test_dataset = datasets.BouncingBallDataset(
            path='./data/bouncing_ball_test.npz'
        )
    elif dataset == '3modesystem':
        train_dataset = datasets.ThreeModeSystemDataset(
            path='./data/3modesystem.npz'
        )
        test_dataset = datasets.ThreeModeSystemDataset(
            path='./data/3modesystem_test.npz'
        )
    elif dataset == 'bee':
        train_dataset = datasets.BeeDataset(
            path='./data/bee.npz'
        )
        test_dataset = datasets.BeeDataset(
            path='./data/bee_test.npz'
        )
    return train_dataset, test_dataset


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
    with open(os.path.join(config['log_dir'], 'config.json'), 'w') as fp:
        json.dump(config, fp)

    # DATA
    train_dataset, test_dataset = get_dataset(config['dataset'])
    num_workers = 0 if config['dataset'] == 'bee' else 4
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'],
        num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=80, pin_memory=True)

    train_gen = iter(train_loader)
    test_gen = iter(test_loader)

    print(f'Running {config["model"]} on {config["dataset"]}.')
    print(f'Train size: {len(train_dataset)}. Test size: {len(test_dataset)}.')

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
        'nmi_score': [],
        'accuracy': [],
        'ari_score': [],
        'f1_score': [],
        'step': []
    }
    for step in range(start_step, config['num_steps'] + 1):
        try:
            train_batch, train_label = next(train_gen)
            train_batch = train_batch.to(device)
        except StopIteration:
            train_gen = iter(train_loader)
        train_result = train_step(
            train_batch, model, optimizer, step, config)

        if step % config['save_steps'] == 0 or step == config['num_steps']:
            model_path = os.path.join(config['model_dir'], f'model_{step}.pt')
            torch.save({'step': step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config},
                       model_path)

        if step % config['log_steps'] == 0 or step == config['num_steps']:
            summary_items = {
                "params/learning_rate": train_result['lr'],
                "params/switch_temperature":
                    train_result['switch_temperature'],
                "params/dur_temperature": train_result['dur_temperature'],
                "params/cross_entropy_coef": train_result['xent_coeff'],
                "elbo/training": train_result[config['objective']],
                "xent/training": train_result['crossent_regularizer']}
            train_result['true_seg'] = train_label
            plot_results(train_result)

            # Plot duration models
            if config['model'] == 'REDSDS':
                dummy_ctrls = torch.ones(1, 1, 1, device=device)
                rho = torch2numpy(
                    model.ctrl2nstf_network.rho(
                        dummy_ctrls,
                        temperature=train_result['dur_temperature']
                    )
                )[0, 0]
                matplotlib_fig = tensorboard_utils.show_duration_dists(
                    fig_size=(15, rho.shape[0] * 2),
                    rho=rho)
                fig_numpy_array = tensorboard_utils.plot_to_image(
                    matplotlib_fig)
                summary.add_image(
                    'Duration', fig_numpy_array, step, dataformats='HWC')

            # Evaluate Segmentation
            if step == config['num_steps']:
                extra_args = dict()
                if config['model'] == 'REDSDS':
                    extra_args = {
                        'dur_temperature': train_result['dur_temperature']
                    }
                true_segs = []
                pred_segs = []
                true_tss = []
                recons_tss = []
                for test_batch, test_label in test_loader:
                    test_batch = test_batch.to(device)
                    test_result = model(
                        test_batch,
                        switch_temperature=train_result['switch_temperature'],
                        num_samples=1,
                        deterministic_inference=True,
                        **extra_args)
                    pred_seg = torch2numpy(
                        torch.argmax(
                            test_result["log_gamma"][0], dim=-1)
                    )
                    true_seg = torch2numpy(
                        test_label[:, :config['context_length']]
                    )
                    true_ts = torch2numpy(test_result["inputs"])
                    recons_ts = torch2numpy(test_result["reconstructions"])
                    true_tss.append(true_ts)
                    recons_tss.append(recons_ts)
                    true_segs.append(true_seg)
                    pred_segs.append(pred_seg)
                true_tss = np.concatenate(true_tss, 0)
                recons_tss = np.concatenate(recons_tss, 0)
                true_segs = np.concatenate(true_segs, 0)
                pred_segs = np.concatenate(pred_segs, 0)
                seg_metrics = evaluate_segmentation(
                    true_segs, pred_segs, K=config['num_categories'])
                all_metrics['step'].append(step)
                for k, v in seg_metrics.items():
                    summary_items[f'metrics/{k}'] = v
                    all_metrics[k].append(v)

            for k, v in summary_items.items():
                summary.add_scalar(k, v, step)
            summary.flush()

    with open(os.path.join(config['log_dir'], 'metrics.json'), 'w') as fp:
        json.dump(all_metrics, fp)
    np.savez(
        os.path.join(config['log_dir'], 'final_results.npz'),
        true_tss=true_tss,
        recons_tss=recons_tss,
        true_segs=true_segs,
        pred_segs=pred_segs
    )

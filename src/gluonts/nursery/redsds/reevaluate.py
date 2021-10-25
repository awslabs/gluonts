import torch
import argparse
from pathlib import Path
import numpy as np
from src.model_utils import build_model
import src.datasets as datasets
import src.evaluation as evaluation
from src.torch_utils import torch2numpy


def get_test_dataset(config, test_path=Path('./data/')):
    if config['experiment'] == 'gts_univariate':
        test_dataset = datasets.GTSUnivariateDataset(
            config['dataset'], batch_size=50, mode='test')
    else:
        if config['dataset'] == '3modesystem':
            test_dataset = datasets.ThreeModeSystemDataset(
                path=str(test_path / f"{config['dataset']}_test.npz")
            )
        elif config['dataset'] == 'bee':
            test_dataset = datasets.BeeDataset(
                path='./data/bee_test.npz'
            )
        elif config['dataset'] == 'bouncing_ball':
            test_dataset = datasets.BouncingBallDataset(
                path='./data/bouncing_ball_test.npz'
            )
        else:
            raise ValueError(f"There is no {config['dataset']} dataset.")

    return test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dev', type=str, default='cuda:0')
    args = parser.parse_args()

    ckpt_file = args.ckpt
    try:
        ckpt = torch.load(ckpt_file, map_location='cpu')
    except FileNotFoundError:
        print(f'Can\'t find {ckpt_file}!')

    config = ckpt['config']
    device = torch.device(args.dev)
    model = build_model(config=config)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    test_dataset = get_test_dataset(config)

    if config['experiment'] == 'gts_univariate':
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=None)

        ctx_len = config['context_length']
        pred_len = config['prediction_length']

        agg_metrics = evaluation.evaluate_gts_dataset(
            test_dataset,
            model,
            max_len=np.inf,
            device=device,
            deterministic_z=False,
            deterministic_x=False,
            deterministic_y=False,
            batch_size=100)

        CRPS = agg_metrics['mean_wQuantileLoss']
        print(config['dataset'], CRPS)

    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=80)
        extra_args = dict()
        if config['model'] == 'REDSDS':
            extra_args = {
                'dur_temperature': 1.0
            }
        true_segs = []
        pred_segs = []
        true_tss = []
        recons_tss = []
        for test_batch, test_label in test_loader:
            test_batch = test_batch.to(device)
            test_result = model(
                test_batch,
                switch_temperature=1.0,
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
        seg_metrics = evaluation.evaluate_segmentation(
            true_segs, pred_segs, K=config['num_categories'])
        print(config['dataset'], seg_metrics)

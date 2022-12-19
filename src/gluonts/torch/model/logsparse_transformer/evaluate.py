from __future__ import unicode_literals, print_function, division
import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    train_seq,
    enc_len,
    dec_len,
    series_id,
    scaling_factor,
    encoder,
    pred_samples,
):
    with torch.no_grad():
        batch_size = train_seq.size(0)
        Pred_series = torch.zeros(
            pred_samples, dec_len, batch_size, 1, device=device
        )
        for i in range(dec_len):
            mu, sigma = encoder(
                series_id, train_seq[:, : (enc_len + 1 + i), :]
            )
            mu = mu[:, -1, :]
            sigma = sigma[:, -1, :]
            Gaussian = torch.distributions.normal.Normal(mu, sigma)
            pred = Gaussian.sample(torch.Size([pred_samples]))
            if i < (dec_len - 1):
                train_seq[:, enc_len + i + 1, 0] = torch.squeeze(mu)
            pred = pred * torch.unsqueeze(
                torch.unsqueeze(scaling_factor, dim=1), dim=0
            )
            Pred_series[:, i, :, :] = pred
        return torch.squeeze(Pred_series)


def evaluateIters(test_loader, enc_len, dec_len, encoder, pred_samples):
    predictions = []
    encoder.eval()
    for id, data in enumerate(test_loader):
        (
            train_seq,
            gt,
            series_id,
            scaling_factor,
        ) = data  # (batch_size,seq, value) --> (seq,batch_size,value)
        train_seq = train_seq.to(torch.float32).to(device)
        gt = gt.to(torch.float32).to(device)
        series_id = series_id.to(device)
        scaling_factor = scaling_factor.to(torch.float32).to(device)
        pred = evaluate(
            train_seq,
            enc_len,
            dec_len,
            series_id,
            scaling_factor,
            encoder,
            pred_samples,
        )
        predictions.append(pred)

    return predictions


def ND_Metrics(predictions, test_loader):
    summation = 0
    diff = 0
    for idx, data in enumerate(test_loader):
        (
            _,
            gt,
            _,
            _,
        ) = data  # (batch_size,seq, value) --> (seq,batch_size,value)
        gt = gt.permute(1, 0).to(torch.float32).to(device)
        series = torch.median(predictions[idx], dim=0)[0]
        diff += torch.sum(torch.abs(torch.squeeze(gt) - torch.squeeze(series)))
        summation += torch.sum(torch.squeeze(torch.abs(gt)))
    return diff.item() / summation.item()


def RMSE_Metrics(predictions, test_loader):
    summation = 0
    sqr_diff = 0
    count_iterms = 0
    for idx, data in enumerate(test_loader):
        _, gt, _, _ = data
        gt = gt.permute(1, 0).to(torch.float32).to(device)
        series = torch.median(predictions[idx], dim=0)[0]
        abs_diff = torch.abs(torch.squeeze(gt) - torch.squeeze(series))
        sqr_diff += torch.sum(abs_diff * abs_diff).item()
        summation += torch.sum(torch.squeeze(torch.abs(gt))).item()
        count_iterms += gt.shape[0] * gt.shape[1]
    return np.sqrt(sqr_diff / count_iterms) / (summation / count_iterms)


def Rou_Risk(rou, predictions, test_loader, pred_samples):
    rou_th = int(pred_samples * (1 - rou))
    numerator = 0
    denominator = 0
    for idx, data in enumerate(test_loader):
        _, gt, _, _ = data
        gt = gt.permute(1, 0).to(torch.float32).to(device)
        rou_pred = torch.topk(predictions[idx], dim=0, k=rou_th)[0][-1, :, :]
        abs_diff = torch.abs(rou_pred - gt)
        numerator += (
            2
            * (
                torch.sum(rou * abs_diff[gt > rou_pred])
                + torch.sum((1 - rou) * abs_diff[gt <= rou_pred])
            ).item()
        )
        denominator += torch.sum(gt).item()
    return numerator / denominator

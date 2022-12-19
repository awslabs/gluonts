from __future__ import unicode_literals, print_function, division
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import torch.nn as nn
import os
from torch import optim
import torch.nn.functional as F
import time
import math
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from model import *
from utils import *
from datasets import *
from evaluate import *
from optimization import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(
    description="PyTorch Transformer on Time series forecasting"
)
parser.add_argument(
    "--path", default="", type=str, help="Time series data path"
)
parser.add_argument(
    "--outdir", default="", type=str, help="The name of saved model"
)
parser.add_argument(
    "--input-size",
    default=5,
    type=int,
    help="input_size (default: 5 = (4 covariates + 1 dim point))",
)
parser.add_argument(
    "--train-ins-num",
    default=500000,
    type=int,
    help="num of train instances (default: 500K)",
)
parser.add_argument(
    "--batch-size", default=64, type=int, help="mini-batch size (default: 64)"
)
parser.add_argument(
    "--eval_batch_size",
    default=-1,
    type=int,
    help="eval_batch_size default is equal to training batch_size",
)
parser.add_argument(
    "--n_head", default=8, type=int, help="n_head (default: 8)"
)
parser.add_argument(
    "--num-layers", default=3, type=int, help="num-layers (default: 3)"
)
parser.add_argument(
    "--epoch", default=20, type=int, help="epoch (default: 20)"
)
parser.add_argument(
    "--embedded_dim",
    default=20,
    type=int,
    help=" The dimention of Position embedding and time series ID embedding",
)
parser.add_argument(
    "--pred-samples",
    default=200,
    type=int,
    help="series samples during prediction (default: 200)",
)
parser.add_argument(
    "--print-freq", default=100, type=int, help="print-freq (default: 100)"
)
parser.add_argument(
    "--lr", default=0.001, type=float, help="initial learning rate"
)
parser.add_argument(
    "--weight_decay", default=0, type=float, help="weight_decay"
)
parser.add_argument("--embd_pdrop", type=float, default=0.1)
parser.add_argument("--attn_pdrop", type=float, default=0.1)
parser.add_argument("--resid_pdrop", type=float, default=0.1)
parser.add_argument(
    "--overlap",
    default=False,
    action="store_true",
    help="If we overlap prediction range during sampling",
)
parser.add_argument(
    "--scale_att", default=False, action="store_true", help="Scaling Attention"
)
parser.add_argument(
    "--sparse",
    default=False,
    action="store_true",
    help="Perform the simulation of sparse attention ",
)
parser.add_argument(
    "--pred_days",
    default=7,
    type=int,
    help="How many days to predict after the train_end date",
)
parser.add_argument("--seed", default=0, type=int, help="set random seed")
parser.add_argument(
    "--enc_len",
    default=168,
    type=int,
    help="The lengh of training range (encoder part)",
)
parser.add_argument(
    "--dec_len",
    default=24,
    type=int,
    help="The lengh of test range (decoder part)",
)
parser.add_argument(
    "--acc_steps",
    default=1,
    type=int,
    help="The steps to accumulate gradient)",
)
parser.add_argument(
    "--dataset", default="", type=str, help="Dataset you want to train"
)
parser.add_argument(
    "--v_partition", default=0.1, type=float, help="validation_partition"
)
parser.add_argument(
    "--q_len", default=1, type=int, help="kernel size for generating key-query"
)
parser.add_argument(
    "--early_stop_ep", default=5, type=int, help="early_stop_ep"
)
parser.add_argument(
    "--pred_samples", default=200, type=int, help="samples of prediction"
)
parser.add_argument(
    "--sub_len", default=1, type=int, help="sub_len of sparse attention"
)
parser.add_argument(
    "--warmup_proportion",
    default=-1,
    type=float,
    help="warmup_proportion for BERT Adam",
)
parser.add_argument(
    "--optimizer", default="Adam", type=str, help="Choice BERTAdam or Adam"
)


def save_checkpoint(state, epoch, loss_is_best, filename):
    torch.save(state, filename + "/" + str(epoch) + "_v_loss.pth.tar")
    if loss_is_best:
        torch.save(state, filename + "/best_v_loss.pth.tar")


def valid(train_seq, gt, series_id, scaling_factor, encoder):
    with torch.no_grad():
        mu, sigma = encoder(series_id, train_seq)
        scaling_factor = torch.unsqueeze(
            torch.unsqueeze(scaling_factor, dim=1), dim=1
        )
        criterion = GaussianLoss(scaling_factor * mu, scaling_factor * sigma)
        gt = torch.unsqueeze(gt, dim=2)
        loss = criterion(gt)

    return loss.item()


def validIters(valid_loader, encoder):
    encoder.eval()
    total_loss = 0
    for id, data in enumerate(valid_loader):
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
        loss = valid(train_seq, gt, series_id, scaling_factor, encoder)
        total_loss += loss
    loss = total_loss / (id + 1)
    return loss


def trainIters(
    args,
    train_loader,
    valid_loader,
    test_loader,
    encoder,
    weight_decay,
    epochs,
    print_every,
    learning_rate,
    outdir,
    pred_samples,
):

    writer = SummaryWriter(outdir)
    start = time.time()
    if args.optimizer == "Adam":
        print("Vanilla Adam")
        encoder_optimizer = optim.Adam(
            encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif args.optimizer == "BERTAdam":
        print("BERTAdam")
        num_train_steps = int(
            len(train_loader) / args.acc_steps * epochs
        )  # This is equal to default Adam.
        encoder_optimizer = BERTAdam(
            encoder.parameters(),
            lr=learning_rate,
            warmup=args.warmup_proportion,
            t_total=num_train_steps,
            weight_decay_rate=args.weight_decay,
        )
    else:
        raise NameError("Currently, we only support Adam and BERTAdam")

    epoch_len = len(train_loader)
    tr_loss = 0
    global_step = 0
    iter_th = 0
    batch_num = len(train_loader)
    n_iters = epochs * batch_num
    print("Start training")
    loss_best = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        encoder.train()
        for step, data in enumerate(train_loader):

            train_seq, gt, series_id, scaling_factor = data
            train_seq = train_seq.to(torch.float32).to(device)
            gt = gt.to(torch.float32).to(device)
            series_id = series_id.to(device)
            scaling_factor = scaling_factor.to(torch.float32).to(device)
            mu, sigma = encoder(series_id, train_seq)
            scaling_factor = torch.unsqueeze(
                torch.unsqueeze(scaling_factor, dim=1), dim=1
            )
            criterion = GaussianLoss(
                scaling_factor * mu, scaling_factor * sigma
            )
            gt = torch.unsqueeze(gt, dim=2)
            loss = criterion(gt)
            loss = loss / args.acc_steps
            loss.backward()
            tr_loss += loss.item()
            iter_th += 1
            if (step + 1) % args.acc_steps == 0:
                if (
                    args.optimizer == "Adam"
                ):  # Since BERTAdam already performs gradient clipping, we only perform gradient clipping for Adam.
                    nn.utils.clip_grad_norm_(encoder.parameters(), 1)
                encoder_optimizer.step()
                encoder.zero_grad()
                global_step += 1
                writer.add_scalar("training/train_loss", tr_loss, global_step)
                tr_loss = 0

            if (iter_th + 1) % (print_every + 1) == 0:
                print(
                    "%s (%d %d%%)"
                    % (
                        timeSince(start, iter_th / n_iters),
                        iter_th,
                        iter_th / n_iters * 100,
                    )
                )

        v_loss = validIters(valid_loader, encoder)
        writer.add_scalar("training/validation_loss", v_loss, epoch + 1)
        if v_loss < loss_best:
            loss_best = v_loss
            loss_is_best = True
            best_epoch = epoch
        else:
            loss_is_best = False

        print("Current v loss: ", v_loss, "Best v loss:", loss_best)
        if epoch - best_epoch >= args.early_stop_ep:
            print("Achieve early_stop_ep and current epoch is", epoch)
            break
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": encoder.state_dict(),
                "optimizer": encoder_optimizer.state_dict(),
            },
            epoch + 1,
            loss_is_best,
            outdir,
        )

    encoder.load_state_dict(
        torch.load(outdir + "/best_v_loss.pth.tar")["state_dict"]
    )
    predictions = evaluateIters(
        test_loader, args.enc_len, args.dec_len, encoder, pred_samples
    )
    ND_metrics = ND_Metrics(predictions, test_loader)
    RMSE_metrics = RMSE_Metrics(predictions, test_loader)
    rou_metrics = Rou_Risk(0.9, predictions, test_loader, pred_samples)
    print("ND: ", ND_metrics)
    print("RMSE: ", RMSE_metrics)
    print("rou-90: ", rou_metrics)

    return 0


def main():
    global args
    args = parser.parse_args()
    np.random.seed(0)  # Fix training and validation set
    num_train = args.train_ins_num
    indices = list(range(num_train))
    split = int(args.v_partition * num_train)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.dataset == "traffic":
        train_set = Traffic(
            args.path,
            args.train_ins_num,
            overlap=args.overlap,
            pred_days=args.pred_days,
            win_len=args.enc_len + args.dec_len,
        )
        test_set = TrafficTest(
            train_set.points,
            train_set.covariates,
            train_set.withhold_len,
            args.enc_len,
            args.dec_len,
        )
    elif args.dataset == "traffic_fine":
        train_set = Traffic_fine(
            args.path,
            args.train_ins_num,
            overlap=args.overlap,
            pred_days=args.pred_days,
            win_len=args.enc_len + args.dec_len,
        )
        test_set = Traffic_fineTest(
            train_set.points,
            train_set.covariates,
            train_set.withhold_len,
            args.enc_len,
            args.dec_len,
        )
    elif args.dataset == "ele":
        train_set = Ele(
            args.path,
            args.train_ins_num,
            overlap=args.overlap,
            pred_days=args.pred_days,
            win_len=args.enc_len + args.dec_len,
        )
        test_set = EleTest(
            train_set.points,
            train_set.covariates,
            train_set.withhold_len,
            args.enc_len,
            args.dec_len,
        )
    elif args.dataset == "ele_fine":
        train_set = Ele_fine(
            args.path,
            args.train_ins_num,
            overlap=args.overlap,
            pred_days=args.pred_days,
            win_len=args.enc_len + args.dec_len,
        )
        test_set = Ele_fineTest(
            train_set.points,
            train_set.covariates,
            train_set.withhold_len,
            args.enc_len,
            args.dec_len,
        )
    elif args.dataset == "m4":
        train_set = M4(
            args.path,
            args.train_ins_num,
            overlap=args.overlap,
            pred_days=args.pred_days,
            win_len=args.enc_len + args.dec_len,
        )
        test_set = M4Test(
            train_set.points,
            train_set.covariates,
            train_set.withhold_len,
            args.enc_len,
            args.dec_len,
        )
    elif args.dataset == "wind":
        train_set = Wind(
            args.path,
            args.train_ins_num,
            overlap=args.overlap,
            pred_days=args.pred_days,
            win_len=args.enc_len + args.dec_len,
        )
        test_set = WindTest(
            train_set.points,
            train_set.covariates,
            train_set.withhold_len,
            args.enc_len,
            args.dec_len,
        )
    elif args.dataset == "solar":
        train_set = Solar(
            args.path,
            args.train_ins_num,
            overlap=args.overlap,
            pred_days=args.pred_days,
            win_len=args.enc_len + args.dec_len,
        )
        test_set = SolarTest(
            train_set.points,
            train_set.covariates,
            train_set.withhold_len,
            args.enc_len,
            args.dec_len,
        )
    else:
        raise NameError(
            "Currently, we only support traffic, ele, m4, solar and wind"
        )
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=train_sampler,
    )
    if args.eval_batch_size == -1:
        eval_batch_size = args.batch_size
    else:
        eval_batch_size = args.eval_batch_size

    valid_loader = DataLoader(
        train_set,
        batch_size=eval_batch_size,
        num_workers=4,
        drop_last=True,
        sampler=valid_sampler,
    )
    test_loader = DataLoader(
        test_set, batch_size=eval_batch_size, num_workers=4
    )

    encoder = DecoderTransformer(
        args,
        input_dim=args.input_size,
        n_head=args.n_head,
        layer=args.num_layers,
        seq_num=train_set.seq_num,
        n_embd=args.embedded_dim,
        win_len=args.enc_len + args.dec_len,
    )
    encoder = nn.DataParallel(encoder).to(device)
    print(args)
    print(encoder)
    trainIters(
        args,
        train_loader,
        valid_loader,
        test_loader,
        encoder,
        args.weight_decay,
        args.epoch,
        print_every=args.print_freq,
        learning_rate=args.lr,
        outdir=args.outdir,
        pred_samples=args.pred_samples,
    )


if __name__ == "__main__":
    main()

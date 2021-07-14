# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


from typing import Dict, Optional, Tuple
from collections.abc import Callable

import numpy as np

import torch
from torch import nn

# import torch.optim as optim
import torch_optimizer as optim

import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, FBeta, ConfusionMatrix

import ncad
from ncad.ts import TimeSeriesDataset
from ncad.model.distances import CosineDistance
from ncad.model.outlier_exposure import coe_batch
from ncad.model.mixup import mixup_batch
from ncad.utils.pl_metrics import CachePredictions
from ncad.utils.donut_metrics import adjust_predicts_donut, adjust_predicts_multiple_ts, best_f1_search_grid


class NCAD( pl.LightningModule ):
    """ Neural Contrastive Detection in Time Series
    """

    def __init__(
            self,
            # hparams for the input data
            ts_channels: int,
            window_length: int,
            suspect_window_length: int,
            # hparams for encoder
            tcn_kernel_size: int,
            tcn_layers: int,
            tcn_out_channels: int,
            tcn_maxpool_out_channels: int = 1,
            embedding_rep_dim: int = 64,
            normalize_embedding: bool = True,
            # hparams for classifier
            distance: nn.Module = CosineDistance(),
            classification_loss: nn.Module = nn.BCELoss(),
            classifier_threshold: float = 0.5,
            threshold_grid_length_val: float = 0.10,
            threshold_grid_length_test: float = 0.05,
            # hparams for batch
            coe_rate: float = 0.0,
            mixup_rate: float = 0.0,
            # hparams for validation and test
            stride_rolling_val_test: Optional[int] = None,
            val_labels_adj:bool = True,
            val_labels_adj_fun: Callable = adjust_predicts_donut,
            test_labels_adj:bool = True,
            test_labels_adj_fun: Callable = adjust_predicts_donut,
            max_windows_unfold_batch: Optional[int] = None,
            # hparams for optimizer
            learning_rate: float = 3e-4,
            *args, **kwargs) -> None:
        
        super().__init__()
        
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        # Encoder Network
        self.encoder = ncad.model.TCNEncoder(
            in_channels = self.hparams.ts_channels,
            out_channels = self.hparams.embedding_rep_dim,
            kernel_size = self.hparams.tcn_kernel_size,
            tcn_channels = self.hparams.tcn_out_channels,
            tcn_layers = self.hparams.tcn_layers,
            tcn_out_channels = self.hparams.tcn_out_channels,
            maxpool_out_channels = self.hparams.tcn_maxpool_out_channels,
            normalize_embedding = self.hparams.normalize_embedding,
        )

        # Contrast Classifier
        self.classifier = ncad.model.ContrastiveClasifier(
            distance = distance,
        )
        
        # Set classification loss
        self.classification_loss = classification_loss
        
        # Label adjustemt for validation
        self.val_labels_adj_fun = val_labels_adj_fun
        self.test_labels_adj_fun = test_labels_adj_fun

        # Define validation metrics
        # NOTE: We don't use torchmetrics directly because
        # we adjust the threshold at every validation step
        val_metrics = dict(
            cache_preds = CachePredictions(compute_on_step=False),
        )
        self.val_metrics = nn.ModuleDict(val_metrics)
        
        # Define test metrics
        # NOTE: We don't use torchmetrics directly because
        # we adjust the consider the best threshold for testing
        test_metrics = dict(
            cache_preds = CachePredictions(compute_on_step=False),
        )
        self.test_metrics = nn.ModuleDict(test_metrics)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # The encoder could manage other window lengths,
        # but all training and validation is currently performed with a single length
        assert x.shape[-1] == self.hparams.window_length

        ts_whole_embedding = self.encoder( x )
        ts_context_embedding = self.encoder( x[...,:-self.hparams.suspect_window_length] )
        
        logits_anomaly = self.classifier(ts_whole_embedding, ts_context_embedding)

        return logits_anomaly

    def training_step(self, batch: Tuple[torch.Tensor,torch.Tensor], batch_idx) -> Dict[str, torch.Tensor]:
        
        x, y = self.xy_from_batch( batch )
        
        if self.hparams.coe_rate > 0:
            x_oe, y_oe = coe_batch(
                x=x, y=y,
                coe_rate = self.hparams.coe_rate,
                suspect_window_length = self.hparams.suspect_window_length,
                random_start_end = True,
            )
            # Add COE to training batch
            x = torch.cat((x,x_oe), dim=0)
            y = torch.cat((y,y_oe), dim=0)

        if self.hparams.mixup_rate > 0.0:
            x_mixup, y_mixup = mixup_batch(
                x=x, y=y,
                mixup_rate = self.hparams.mixup_rate,
            )
            # Add Mixup to training batch
            x = torch.cat((x,x_mixup), dim=0)
            y = torch.cat((y,y_mixup), dim=0)
            
        # Compute predictions
        logits_anomaly = self(x).squeeze()
        probs_anomaly = torch.sigmoid( logits_anomaly )
        
        # Calculate Loss
        loss = self.classification_loss( probs_anomaly, y)
        
        assert torch.isfinite(loss).item()

        
        # Logging loss
        self.log('train_loss_step', loss, on_step = True, on_epoch = False, prog_bar = True, logger = True)

        return {'loss':loss}

    def on_validation_epoch_start(self):
        # Reset all states in validation metrics
        for key in self.val_metrics.keys():
            self.val_metrics[key].reset()

    def validation_step(self, batch, batch_idx):
        x, y = self.xy_from_batch( batch )
        
        # Compute predictions
        probs_anomaly, _ = self.detect(
            ts = x,
            threshold_prob_vote = self.hparams.classifier_threshold,
            stride = int(self.hparams.stride_rolling_val_test) if self.hparams.stride_rolling_val_test else self.hparams.suspect_window_length,
        )

        # Eliminate timesteps with nan's in prediction
        nan_time_idx = torch.isnan(probs_anomaly).int().sum(dim=0).bool()
        y = y[:,~nan_time_idx]
        probs_anomaly = probs_anomaly[:,~nan_time_idx]
        target = y
        
        self.val_metrics['cache_preds'](preds=probs_anomaly, target=target)
        
    def on_validation_epoch_end(self):
        stage = 'val'
        ### Compute metrics and find "best" threshold
        score, target = self.val_metrics['cache_preds'].compute()
        
        # score, target into lists of 1-d np.ndarray
        score_np, target_np = [], []
        for i in range(len(score)):
            score_i = score[i].cpu().numpy()
            target_i = target[i].cpu().numpy()
            assert score_i.shape[0]==1, 'Expected 1-d array with the predictad labels of the TimeSeries'
            assert target_i.shape[0]==1, 'Expected 1-d array with the observed labels of the TimeSeries'
            score_np.append( score_i[0,:] )
            target_np.append( target_i[0,:] )

        # Criteria: best F1
        metrics_best, threshold_best = best_f1_search_grid(
            score=score_np,
            target=target_np,
            adjust_predicts_fun = self.val_labels_adj_fun if self.hparams.val_labels_adj else None,
            threshold_values = np.round( np.arange(0.0, 1.0, self.hparams.threshold_grid_length_val ), decimals=5),
            # threshold_bounds = [0.01, 0.99],
        )
        
        self.hparams.classifier_threshold = threshold_best
        self.log(f"classifier_threshold", self.hparams.classifier_threshold, prog_bar = True, logger = True)

        # Log metrics
        for key, value in metrics_best.items():
            self.log(f"{stage}_{key}", value, prog_bar = True if key=='f1' else False, logger = True)
    
    def test_step(self, batch, batch_idx):
        
        x, y = self.xy_from_batch( batch )
        
        # Compute predictions
        probs_anomaly, _ = self.detect(
            ts = x,
            threshold_prob_vote = self.hparams.classifier_threshold,
            stride = int(self.hparams.stride_rolling_val_test) if self.hparams.stride_rolling_val_test else self.hparams.suspect_window_length,
        )

        # Eliminate timesteps with nan's in prediction
        nan_time_idx = torch.isnan(probs_anomaly).int().sum(dim=0).bool()
        y = y[:,~nan_time_idx]
        probs_anomaly = probs_anomaly[:,~nan_time_idx]
        target = y
        
        self.test_metrics['cache_preds'](preds=probs_anomaly, target=target)
        
    def on_test_epoch_end(self):
        stage = 'test'
        ### Compute metrics and find "best" threshold
        score, target = self.test_metrics['cache_preds'].compute()
        
        # score, target into lists of 1-d np.ndarray
        score_np, target_np = [], []
        for i in range(len(score)):
            score_i = score[i].cpu().numpy()
            target_i = target[i].cpu().numpy()
            assert score_i.shape[0]==1, 'Expected 1-d array with the predictad labels of the TimeSeries'
            assert target_i.shape[0]==1, 'Expected 1-d array with the observed labels of the TimeSeries'
            score_np.append( score_i[0,:] )
            target_np.append( target_i[0,:] )
        
        # Criteria: best F1
        metrics_best, threshold_best = best_f1_search_grid(
            score=score_np,
            target=target_np,
            adjust_predicts_fun = self.test_labels_adj_fun if self.hparams.test_labels_adj else None,
            threshold_values = np.round( np.arange(0.0, 1.0, self.hparams.threshold_grid_length_test ), decimals=5),
            # threshold_bounds = [0.01, 0.99],
        )
        
        self.hparams.classifier_threshold = threshold_best
        self.log(f"classifier_threshold", self.hparams.classifier_threshold, prog_bar = True, logger = True)

        # Log metrics
        for key, value in metrics_best.items():
            self.log(f"{stage}_{key}", value, prog_bar = True if key=='f1' else False, logger = True)
    
    def configure_optimizers(self):
        # optim_class = optim.Adam
        optim_class = optim.Yogi
        # optim_class = optim.AdaBound

        optimizer = optim_class( self.parameters(), lr = self.learning_rate )

        return optimizer

    def detect(
            self,
            ts: torch.Tensor,
            threshold_prob_vote: float = 0.5,
            stride: Optional[int] = None,
            *args, **kwargs) -> torch.Tensor:
        
        """ Deploys the model over a tensor representing the time series

        Args:
            ts: Tensor with the time series. Shape (batch_size, ts_channels, time)
        
        Output
            pred: Tensor with the estimated probability of each timestep being anomalous. Shape (batch_size, time)
        """

        assert (0<=threshold_prob_vote<=1)

        if stride is None:
            stride = self.hparams.suspect_window_length
        
        batch_size, ts_channels, T = ts.shape
        num_windows = int(1+(T-self.hparams.window_length)/stride)

        # Define functions for folding and unfolding the time series
        unfold_layer = nn.Unfold(kernel_size=(ts_channels, self.hparams.window_length), stride=stride)
        fold_layer = nn.Fold(output_size=(1,T), kernel_size=(1,self.hparams.window_length), stride=stride)
        
        # Currently, only 4-D input tensors (batched image-like tensors) are supported
        # images = (batch, channels, height, width)
        # we adapt our time series creating a height channel of dimension 1, and then 
        ts_windows = unfold_layer(ts.unsqueeze(1))
        
        assert ts_windows.shape == (batch_size, ts_channels * self.hparams.window_length, num_windows )

        ts_windows = ts_windows.transpose(1,2)
        ts_windows = ts_windows.reshape(batch_size, num_windows, ts_channels, self.hparams.window_length)

        # Also posible via tensor method
        # ts_windows = ts.unfold(dimension=1,size=self.hparams.window_length,step=stride)

        with torch.no_grad():
            if self.hparams.max_windows_unfold_batch is None:
                logits_anomaly = self( ts_windows.flatten(start_dim=0, end_dim=1) )
            else:
                # For very long time series, it is neccesary to process the windows in smaller chunks
                logits_anomaly = [ self(ts_windows_chunk) for ts_windows_chunk in torch.split( ts_windows.flatten(start_dim=0, end_dim=1), self.hparams.max_windows_unfold_batch, dim=0 )]
                logits_anomaly = torch.cat(logits_anomaly, dim=0)

        # Check model output shape: one label per (multivariate) window
        assert logits_anomaly.shape == (batch_size*num_windows, 1)
        
        # Repeat prediction for all timesteps in the suspect window, and reshape back before folding
        logits_anomaly = logits_anomaly.reshape(batch_size, num_windows, 1)
        logits_anomaly = logits_anomaly.repeat(1,1,self.hparams.window_length)
        logits_anomaly[...,:-self.hparams.suspect_window_length] = np.nan
        logits_anomaly = logits_anomaly.transpose(1,2)

        assert logits_anomaly.shape == (batch_size, self.hparams.window_length, num_windows)
        
        # Function to squeeze dimensions 1 and 2 after folding
        squeeze_fold = lambda x: x.squeeze(2).squeeze(1)

        ### Count the number of predictions per timestep ###
        # Indicates entries in logits_anomaly with a valid prediction
        id_suspect = torch.zeros_like(logits_anomaly)
        id_suspect[:,-self.hparams.suspect_window_length:] = 1.0
        num_pred = squeeze_fold( fold_layer(id_suspect) )

        # Average of predicted probability of being anomalous for each timestep
        anomaly_probs = torch.sigmoid( logits_anomaly )
        # anomaly_probs_avg = squeeze_fold( fold_layer( anomaly_probs ) ) / num_pred
        anomaly_probs_nanto0 = torch.where(id_suspect==1,anomaly_probs,torch.zeros_like(anomaly_probs))
        anomaly_probs_avg = fold_layer( anomaly_probs_nanto0 ).squeeze(2).squeeze(1) / num_pred
        
        assert anomaly_probs_avg.shape == (batch_size, T)

        # Majority vote
        anomaly_votes = squeeze_fold( fold_layer(1.0*(anomaly_probs>threshold_prob_vote)) )
        anomaly_vote = 1.0*(anomaly_votes > (num_pred/2))

        assert anomaly_vote.shape == (batch_size, T)

        return anomaly_probs_avg, anomaly_vote
    
    def tsdetect(
            self,
            ts_dataset: TimeSeriesDataset,
            stride: Optional[int] = None,
            threshold_prob_vote: float = 0.5,
            *args, **kwargs) -> TimeSeriesDataset:
        """ Deploys the model over a TimeSeriesDataset

        Args:
            ts_dataset: TimeSeriesDataset with the univariate time series.
        
        Output
            pred: Tensor with the estimated probability of each timestep being anomalous. Shape (batch, time)
        """
        
        assert not ts_dataset.nan_ts_values

        # Number of TimeSeries in the dataset
        N = len(ts_dataset)

        # Lengths of each TimeSeries
        ts_lengths = np.asarray([ts.shape[0] for ts in ts_dataset])
        same_length = np.all(ts_lengths == ts_lengths[0])
        
        ts_dataset_out = ts_dataset.copy()
        if same_length:
            # Stack all series and predict
            ts_torch = torch.stack([torch.tensor(ts.values.reshape(ts.shape), device=self.device) for ts in ts_dataset], dim=0)
            ts_torch = ts_torch.transpose(1,2)
            anomaly_probs_avg, anomaly_vote = self.detect(
                ts=ts_torch,
                threshold_prob_vote=threshold_prob_vote,
                stride=stride
            )
            anomaly_probs_avg = anomaly_probs_avg.cpu().numpy()
            anomaly_vote = anomaly_vote.cpu().numpy()
            # # Save prediction in dataset
            # for i, ts in enumerate(ts_dataset_out):
            #     ts.anomaly_probs_avg = anomaly_probs_avg
            #     ts.anomaly_vote = anomaly_vote
                # ts.labels = pred[i].squeeze().numpy()
        else:
            # Predict and save prediction in dataset
            anomaly_probs_avg, anomaly_vote = [],[]
            for i, ts in enumerate(ts_dataset_out):
                ts_torch = torch.tensor(ts.values, device=self.device).reshape(ts.shape).T.unsqueeze(0)
                if ts_torch.dim()==2:
                    ts_torch.unsqueeze(1)
                anomaly_probs_avg_i, anomaly_vote_i = self.detect(
                    ts=ts_torch,
                    threshold_prob_vote=threshold_prob_vote,
                    stride=stride
                )
                anomaly_probs_avg.append(anomaly_probs_avg_i.cpu().numpy())
                anomaly_vote.append(anomaly_vote_i.cpu().numpy())
                # ts.labels = pred.squeeze().numpy()

        return anomaly_probs_avg, anomaly_vote
    
    @staticmethod
    def xy_from_batch(
            batch:Tuple[torch.Tensor]
            ) -> Tuple[torch.Tensor]:
        """ Fit batch dimensions for training and validation

        Args:
            batch : Tuple (x,y) generated by a dataloader (CroppedTimeSeriesDatasetTorch or TimeSeriesDatasetTorch)
                which provides x of shape (batch, number of crops, ts channels, time), and y of shape (batch, number of crops)
        
        This function flatten the first two dimensions: batch, ts sample.
        """
        
        x, y = batch

        # flatten first two dimensions
        if x.dim()==4 and y.dim()==2:
            x = torch.flatten(x, start_dim=0, end_dim=1)
            y = torch.flatten(y, start_dim=0, end_dim=1)
        
        return x, y

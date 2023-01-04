import pytorch_lightning as pl
import torch.nn as nn


class TemporalFusionTransformerLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float, patience: int):
        super().__init__()
        self.model = model
        self.lr = lr
        self.patience = patience

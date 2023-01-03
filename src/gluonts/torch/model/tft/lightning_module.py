import torch.nn as nn


class TemporalFusionTransformerLightningModule:
    def __init__(self, model: nn.Module, lr: float, patience: int):
        self.model = model

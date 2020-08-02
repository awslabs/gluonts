import torch
from torch import nn
from utils.utils import batch_diag_matrix


class BatchDiagMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return batch_diag_matrix(x)

import time
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import random


class Trainer:
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_workers: int = 4,
        pin_memory: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        device: Optional[torch.device] = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __call__(
        self, net: nn.Module, input_names: List[str], data_loader: DataLoader
    ) -> None:
        pass

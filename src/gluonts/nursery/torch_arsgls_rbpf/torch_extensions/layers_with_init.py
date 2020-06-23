import math

from torch import nn
from torch.nn import init


class Conv2d(nn.Conv2d):
    pass
    # def reset_parameters(self):
    #     init.xavier_uniform_(self.weight, gain=1.0)
    #     if self.bias is not None:
    #         init.zeros_(self.bias)


class Linear(nn.Linear):
    pass
    # def reset_parameters(self):
    #     init.xavier_uniform_(self.weight, gain=1.0)
    #     if self.bias is not None:
    #         init.zeros_(self.bias)


class LSTM(nn.LSTM):
    pass
    # def reset_parameters(self):
    #     for name, weight in self.named_parameters():
    #         if "bias" in name:
    #             init.zeros_(weight)
    #         else:
    #             init.xavier_uniform_(weight)

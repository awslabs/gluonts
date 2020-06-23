import torch
from torch import nn
from torch_extensions.mlp import MLP
from utils.utils import one_hot
from dataclasses import dataclass


@dataclass
class ControlInputs:
    state: torch.Tensor
    obs: torch.Tensor
    switch: torch.Tensor

    def __getitem__(self, item):
        return ControlInputs(self.state[item], self.obs[item],
                             self.switch[item])

    def __len__(self):
        assert len(self.state) == len(self.obs) == len(self.switch)
        return len(self.state)


class InputTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u_static_cat, u_time) -> ControlInputs:
        raise NotImplementedError("child classes must implement this")


class InputTransformEmbedder(InputTransformer):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.dims.staticfeat,
            embedding_dim=config.dims.cat_embedding,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputs:
        u = torch.cat([self.embedding(u_static_cat), u_time], dim=-1)
        return ControlInputs(state=u, obs=u, switch=u)


class InputTransformOneHotMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.dims.staticfeat
        self.mlp = MLP(
            dim_in=config.dims.timefeat + config.dims.staticfeat,
            dims_hidden=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputs:
        u_staticfeat_onehot = one_hot(u_static_cat,
                                      num_classes=self.num_classes).to(
            dtype=u_time.dtype)
        u = self.mlp(torch.cat((u_staticfeat_onehot, u_time), dim=-1))
        return ControlInputs(state=u, obs=u, switch=u)


class InputTransformMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(
            dim_in=config.dims.timefeat + config.dims.staticfeat,
            dims_hidden=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputs:
        u = self.mlp(torch.cat((u_static_cat, u_time), dim=-1))
        return ControlInputs(state=u, obs=u, switch=u)


class InputTransformEmbeddingAndMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.dims.staticfeat,
            embedding_dim=config.dims.cat_embedding,
        )
        self.mlp = MLP(
            dim_in=config.dims.cat_embedding + config.dims.timefeat,
            dims_hidden=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputs:
        u = self.mlp(
            torch.cat([self.embedding(u_static_cat), u_time * 5], dim=-1))
        return ControlInputs(state=u, obs=u, switch=u)


class InputTransformEmbeddingAndMLPKVAE(InputTransformEmbeddingAndMLP):
    def forward(self, u_static_cat, u_time) -> ControlInputs:
        u = super().forward(u_static_cat=u_static_cat, u_time=u_time)
        u.obs = None
        u.switch = None
        return u

import torch
from torch import nn

from models_new_will_replace.sgls_rbpf import ControlInputsSGLS
from torch_extensions.mlp import MLP
from utils.utils import one_hot


class InputTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u_static_cat, u_time) -> ControlInputsSGLS:
        raise NotImplementedError("child classes must implement this")


class InputTransformEmbedder(InputTransformer):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.dims.staticfeat,
            embedding_dim=config.dims.cat_embedding,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputsSGLS:
        u = torch.cat([self.embedding(u_static_cat), u_time], dim=-1)
        return ControlInputsSGLS(state=u, target=u, switch=u)


class InputTransformOneHotMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.dims.staticfeat
        self.mlp = MLP(
            dim_in=config.dims.timefeat + config.dims.staticfeat,
            dims=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputsSGLS:
        u_staticfeat_onehot = one_hot(
            u_static_cat, num_classes=self.num_classes
        ).to(dtype=u_time.dtype)
        u = self.mlp(torch.cat((u_staticfeat_onehot, u_time), dim=-1))
        return ControlInputsSGLS(state=u, target=u, switch=u, encoder=u)


class InputTransformMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(
            dim_in=config.dims.timefeat + config.dims.staticfeat,
            dims=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputsSGLS:
        u = self.mlp(torch.cat((u_static_cat, u_time), dim=-1))
        return ControlInputsSGLS(state=u, target=u, switch=u, encoder=u)


class InputTransformEmbeddingAndMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.dims.staticfeat,
            embedding_dim=config.dims.cat_embedding,
        )
        self.mlp = MLP(
            dim_in=config.dims.cat_embedding + config.dims.timefeat,
            dims=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(self, u_static_cat, u_time) -> ControlInputsSGLS:
        # TODO: must now repeat static feats on first dim (after embed).
        #  previously had that in preprocessing.
        u = self.mlp(
            torch.cat([self.embedding(u_static_cat), u_time * 5], dim=-1)
        )
        return ControlInputsSGLS(state=u, target=u, switch=u, encoder=u)


class InputTransformEmbeddingAndMLPKVAE(InputTransformEmbeddingAndMLP):
    def forward(self, u_static_cat, u_time) -> ControlInputsSGLS:
        u = super().forward(u_static_cat=u_static_cat, u_time=u_time)
        u.target = None
        u.switch = None
        u.encoder = None
        return u

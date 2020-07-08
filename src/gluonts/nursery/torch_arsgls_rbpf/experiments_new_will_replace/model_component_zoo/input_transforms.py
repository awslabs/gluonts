from typing import Optional
import torch
from torch import nn
from torch_extensions.mlp import MLP
from utils.utils import one_hot
from dataclasses import dataclass


@dataclass
class ControlInputs:
    state: torch.Tensor
    target: torch.Tensor
    switch: torch.Tensor
    encoder: torch.Tensor

    def __getitem__(self, item):
        return ControlInputs(
            self.state[item],
            self.target[item],
            self.switch[item],
            self.encoder[item]
        )

    def __len__(self):
        assert len(self.state) == len(self.target) == len(self.switch) \
               == len(self.encoder)
        return len(self.state)


class InputTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            feat_static_cat: torch.Tensor,
            time_feat: torch.Tensor,
            seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> ControlInputs:
        raise NotImplementedError("child classes must implement this")

    def _repeat_static_feats(self, feat_static, n_timesteps):
        return feat_static[None, ...].repeat(
            (n_timesteps,) + (1,) * feat_static.ndim)


class InputTransformEmbedder(InputTransformer):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.dims.staticfeat,
            embedding_dim=config.dims.cat_embedding,
        )

    def forward(
            self,
            feat_static_cat: torch.Tensor,
            time_feat: torch.Tensor,
            seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> ControlInputs:
        feat_static_embed_repeat = self._repeat_static_feats(
            feat_static=self.embedding(
                feat_static_cat.squeeze(dim=-1).to(torch.int64),
            ),
            n_timesteps=len(time_feat),
        )
        feat = torch.cat([feat_static_embed_repeat, time_feat], dim=-1)
        return ControlInputs(state=feat, target=feat, switch=feat, encoder=feat)


class InputTransformOneHotMLP(InputTransformer):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.dims.staticfeat
        self.mlp = MLP(
            dim_in=config.dims.timefeat + config.dims.staticfeat,
            dims=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(
            self,
            feat_static_cat: torch.Tensor,
            time_feat: torch.Tensor,
            seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> ControlInputs:
        feat_static_onehot_repeat = self._repeat_static_feats(
            feat_static=one_hot(
                feat_static_cat, num_classes=self.num_classes,
            ).to(dtype=feat_static_cat.dtype),
            n_timesteps=len(time_feat),
        )
        feat = self.mlp(
            torch.cat((feat_static_onehot_repeat, time_feat), dim=-1),
        )
        return ControlInputs(state=feat, target=feat, switch=feat, encoder=feat)


class InputTransformMLP(InputTransformer):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(
            dim_in=config.dims.timefeat + config.dims.staticfeat,
            dims=config.input_transform_dims,
            activations=config.input_transform_activations,
        )

    def forward(
            self,
            feat_static_cat: torch.Tensor,
            time_feat: torch.Tensor,
            seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> ControlInputs:
        feat_static_repeat = self._repeat_static_feats(
            feat_static=feat_static_cat,
            n_timesteps=len(time_feat),
        )
        feat = self.mlp(
            torch.cat((feat_static_repeat, time_feat), dim=-1),
        )
        return ControlInputs(state=feat, target=feat, switch=feat, encoder=feat)


class InputTransformEmbeddingAndMLP(InputTransformer):
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
        # TODO: hard-coded. Hopefully not needed anymore later
        self.time_feat_factor = 5.0

    def forward(
            self,
            feat_static_cat: torch.Tensor,
            time_feat: torch.Tensor,
            seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> ControlInputs:
        feat_static_embed_repeat = self._repeat_static_feats(
            feat_static=self.embedding(
                feat_static_cat.squeeze(dim=-1).to(torch.int64),
            ),
            n_timesteps=len(time_feat),
        )
        feat = self.mlp(
            torch.cat([feat_static_embed_repeat,
                       time_feat * self.time_feat_factor], dim=-1),
        )
        return ControlInputs(state=feat, target=feat, switch=feat, encoder=feat)


class InputTransformEmbeddingAndMLPKVAE(InputTransformEmbeddingAndMLP):
    def forward(
            self,
            feat_static_cat: torch.Tensor,
            time_feat: torch.Tensor,
            seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> ControlInputs:
        u = super().forward(
            feat_static_cat=feat_static_cat,
            time_feat=time_feat,
            seasonal_indicators=seasonal_indicators,
        )
        u.target = None
        u.switch = None
        u.encoder = None
        return u

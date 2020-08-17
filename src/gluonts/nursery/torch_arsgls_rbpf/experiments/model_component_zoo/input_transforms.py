from typing import Optional
import torch
from torch import nn
from torch_extensions.mlp import MLP
from utils.utils import one_hot
from models.base_gls import ControlInputs
from models.sgls_rbpf import (
    ControlInputsSGLS,
    ControlInputsSGLSISSM,
)


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

    def _repeat_static_feats(
        self, feat_static: torch.Tensor, n_timesteps: int
    ):
        return feat_static[None, ...].repeat(
            (n_timesteps,) + (1,) * feat_static.ndim
        )

    def _all_same_controls(
        self,
        ctrl_features: torch.Tensor,
        seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> (ControlInputsSGLS, ControlInputsSGLSISSM):
        if seasonal_indicators is None:
            return ControlInputsSGLS(
                state=ctrl_features,
                target=ctrl_features,
                switch=ctrl_features,
                encoder=ctrl_features,
            )
        else:
            return ControlInputsSGLSISSM(
                state=ctrl_features,
                target=ctrl_features,
                switch=ctrl_features,
                encoder=ctrl_features,
                seasonal_indicators=seasonal_indicators,
            )


class NoControlsDummyInputTransformer(InputTransformer):
    def __init__(self, config=None):
        super().__init__()

    def forward(
        self,
        feat_static_cat: Optional[torch.Tensor] = None,
        time_feat: Optional[torch.Tensor] = None,
        seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> (ControlInputsSGLS, ControlInputsSGLSISSM):
        assert feat_static_cat is None
        assert time_feat is None
        assert seasonal_indicators is None
        return self._all_same_controls(
            ctrl_features=None, seasonal_indicators=None,
        )


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
    ) -> (ControlInputsSGLS, ControlInputsSGLSISSM):
        feat_static_embed_repeat = self._repeat_static_feats(
            feat_static=self.embedding(
                feat_static_cat.squeeze(dim=-1).to(torch.int64),
            ),
            n_timesteps=len(time_feat),
        )
        ctrl_features = torch.cat(
            [feat_static_embed_repeat, time_feat], dim=-1
        )
        return self._all_same_controls(
            ctrl_features=ctrl_features,
            seasonal_indicators=seasonal_indicators,
        )


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
    ) -> (ControlInputsSGLS, ControlInputsSGLSISSM):
        feat_static_onehot_repeat = self._repeat_static_feats(
            feat_static=one_hot(
                feat_static_cat, num_classes=self.num_classes,
            ).to(dtype=feat_static_cat.dtype),
            n_timesteps=len(time_feat),
        )
        ctrl_features = self.mlp(
            torch.cat((feat_static_onehot_repeat, time_feat), dim=-1),
        )
        return self._all_same_controls(
            ctrl_features=ctrl_features,
            seasonal_indicators=seasonal_indicators,
        )


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
    ) -> (ControlInputsSGLS, ControlInputsSGLSISSM):
        feat_static_repeat = self._repeat_static_feats(
            feat_static=feat_static_cat, n_timesteps=len(time_feat),
        )
        ctrl_features = self.mlp(
            torch.cat((feat_static_repeat, time_feat), dim=-1),
        )
        return self._all_same_controls(
            ctrl_features=ctrl_features,
            seasonal_indicators=seasonal_indicators,
        )


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
    ) -> (ControlInputsSGLS, ControlInputsSGLSISSM):
        feat_static_embed_repeat = self._repeat_static_feats(
            feat_static=self.embedding(
                feat_static_cat.squeeze(dim=-1).to(torch.int64),
            ),
            n_timesteps=len(time_feat),
        )
        ctrl_features = self.mlp(
            torch.cat(
                [feat_static_embed_repeat, time_feat * self.time_feat_factor],
                dim=-1,
            ),
        )
        return self._all_same_controls(
            ctrl_features=ctrl_features,
            seasonal_indicators=seasonal_indicators,
        )


class InputTransformEmbeddingAndMLPKVAE(InputTransformEmbeddingAndMLP):
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        time_feat: torch.Tensor,
        seasonal_indicators: Optional[torch.Tensor] = None,
    ) -> (ControlInputsSGLS, ControlInputsSGLSISSM):
        controls = super().forward(
            feat_static_cat=feat_static_cat,
            time_feat=time_feat,
            seasonal_indicators=seasonal_indicators,
        )
        controls.target = None
        controls.switch = None
        controls.encoder = None
        return controls

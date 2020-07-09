from typing import List
import torch
from torch import nn
import numpy as np

from torch_extensions.distributions.parametrised_distribution import (
    prepend_batch_dims,
)


class ParametrisedConditionalDistribution(nn.Module):
    def __init__(
        self,
        stem: nn.Module,
        dist_params: nn.ModuleDict,
        dist_cls,
        n_feature_dims=1,
        allow_cat_inputs=False,
    ):
        super().__init__()
        self.n_feature_dims = n_feature_dims
        self.stem = stem
        self.dist_params = dist_params
        self.dist_cls = dist_cls
        self.allow_cat_inputs = allow_cat_inputs

    def forward(
        self,
        x: (torch.Tensor, tuple, list, set),
        batch_shape_to_prepend=tuple(),
    ):
        x = self._maybe_cat(x)
        batch_shp = x.shape[: -self.n_feature_dims]

        h = self.stem(self._flatten_batch_dims(x, batch_shp=batch_shp))
        dist_params = self.dist_params
        dist_params = {
            name: self._unflatten_batch_dims(param(h), batch_shp=batch_shp)
            for name, param in dist_params.items()
        }
        dist_params = (
            {
                name: prepend_batch_dims(
                    tensor=param, shp=batch_shape_to_prepend
                )
                for name, param in dist_params.items()
            }
            if len(batch_shape_to_prepend) > 0
            else dist_params
        )
        dist = self.dist_cls(**dist_params)
        return dist

    def _flatten_batch_dims(self, x, batch_shp):
        feature_shp = x.shape[-self.n_feature_dims :]
        if len(batch_shp) == 1:
            x_flat = x
        elif len(batch_shp) > 1:
            x_flat = torch.reshape(
                x, shape=(np.prod(tuple(batch_shp)),) + feature_shp
            )
        else:
            raise Exception(f"unexpected batch_shp: {batch_shp}")
        return x_flat

    def _unflatten_batch_dims(self, x_flat, batch_shp):
        feature_shp = x_flat.shape[1:]
        if len(batch_shp) == 1:
            x = x_flat
        elif len(batch_shp) > 1:
            x = torch.reshape(x_flat, shape=batch_shp + feature_shp)
        else:
            raise Exception(f"unexpected batch_shp: {batch_shp}")
        return x

    def _maybe_cat(self, x: (torch.Tensor, tuple, list, set)):
        if isinstance(x, (tuple, list, set)):
            if self.allow_cat_inputs:
                x = torch.cat(x, dim=-1)
            else:
                raise ValueError(
                    f"got input of type {type(x)}, "
                    f"but concat is not set to allowed."
                    f"This feature is for your own safety kid."
                )
        else:
            pass
        return x


class LadderParametrisedConditionalDistribution(
    ParametrisedConditionalDistribution
):
    def __init__(
        self,
        stem: nn.ModuleList,
        dist_params: nn.ModuleList,
        dist_cls,
        n_feature_dims=1,
        allow_cat_inputs=False,
    ):
        super().__init__(
            stem=stem,
            dist_params=dist_params,
            dist_cls=dist_cls,
            n_feature_dims=n_feature_dims,
            allow_cat_inputs=allow_cat_inputs,
        )
        assert len(stem) == len(dist_params)
        # TODO: test this for > 2 hierarchies.
        assert (
            len(stem) == 2
        ), "temporary assertion as we have only 2 hierarchies atm"

    def forward(
        self,
        x: (torch.Tensor, tuple, list, set),
        batch_shape_to_prepend=tuple(),
    ) -> List[torch.distributions.Distribution]:
        x = self._maybe_cat(x)
        batch_shp = x.shape[: -self.n_feature_dims]

        h = x
        dists = []
        for idx_hierarchy, (stem, dist_params, dist_cls) in enumerate(
            zip(self.stem, self.dist_params, self.dist_cls)
        ):
            h = stem(self._flatten_batch_dims(h, batch_shp=batch_shp))
            dist_params = {
                name: self._unflatten_batch_dims(param(h), batch_shp=batch_shp)
                for name, param in dist_params.items()
            }
            dist_params = (
                {
                    name: prepend_batch_dims(
                        tensor=param, shp=batch_shape_to_prepend
                    )
                    for name, param in dist_params.items()
                }
                if len(batch_shape_to_prepend) > 0
                else dist_params
            )
            dist = dist_cls(**dist_params)
            dists.append(dist)
        return dists

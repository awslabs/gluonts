# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import Callable, List, Optional  # noqa: F401

# Third-party imports
import mxnet.gluon.nn as nn
import numpy as np

# First-party imports
from gluonts.core.component import DType, validated
from gluonts.model.common import Tensor


class FeatureEmbedder(nn.HybridBlock):
    """
    Embed a sequence of categorical features.

    Parameters
    ----------
    cardinalities
        cardinality for each categorical feature.

    embedding_dims
        number of dimensions to embed each categorical feature.
    """

    @validated()
    def __init__(
        self, cardinalities: List[int], embedding_dims: List[int], **kwargs
    ) -> None:
        super().__init__(**kwargs)

        assert (
            len(cardinalities) > 0
        ), "Length of `cardinalities` list must be greater than zero"
        assert len(cardinalities) == len(
            embedding_dims
        ), "Length of `embedding_dims` and `embedding_dims` should match"
        assert all(
            [c > 0 for c in cardinalities]
        ), "Elements of `cardinalities` should be > 0"
        assert all(
            [d > 0 for d in embedding_dims]
        ), "Elements of `embedding_dims` should be > 0"

        self.__num_features = len(cardinalities)

        def create_embedding(i: int, c: int, d: int) -> nn.Embedding:
            embedding = nn.Embedding(c, d, prefix=f'cat_{i}_embedding_')
            self.register_child(embedding)
            return embedding

        with self.name_scope():
            self.__embedders = [
                create_embedding(i, c, d)
                for i, (c, d) in enumerate(zip(cardinalities, embedding_dims))
            ]

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, features: Tensor) -> Tensor:
        """

        Parameters
        ----------
        F

        features
            Categorical features with shape: (N,T,C) or (N,C), where C is the
            number of categorical features.

        Returns
        -------
        concatenated_tensor: Tensor
            Concatenated tensor of embeddings whth shape: (N,T,C) or (N,C),
            where C is the sum of the embedding dimensions for each categorical
            feature, i.e. C = sum(self.config.embedding_dims).
        """

        if self.__num_features > 1:
            # we slice the last dimension, giving an array of length self.__num_features with shape (N,T) or (N)
            cat_feature_slices = F.split(
                features, axis=-1, num_outputs=self.__num_features
            )
        else:
            # F.split will iterate over the second-to-last axis if the last axis is one
            cat_feature_slices = [features]

        return F.concat(
            *[
                embed(F.squeeze(cat_feature_slice, axis=-1))
                for embed, cat_feature_slice in zip(
                    self.__embedders, cat_feature_slices
                )
            ],
            dim=-1,
        )


class FeatureAssembler(nn.HybridBlock):
    """
    Assemble features into an MXNet tensor. Input features are distinguished based on the following criteria:

    - static (time-independent) features vs dynamic (that is, time-dependent)
    - categorical vs real-valued features.

    Dynamic features have shape `(N, T, C)` and static features have shape `(N, C)`, where

    - `N` is the number of elements in the processed batch,
    - `T` is the time dimension,
    - `C` is the number of features.

    If multiple feature types are used, the :class:`FeatureAssembler` will assume that the N and T dimensions
    are the same for all passed arguments.

    Categorical features can be optionally embedded using trained embedding layers via nested :class:`FeatureEmbedder`
    components.

    >>> # noinspection PyTypeChecker
    ... embed_static = FeatureEmbedder(
    ...     cardinalities=[2],
    ...     embedding_dims=[3],
    ...     prefix='embed_static_',
    ... )
    >>> # noinspection PyTypeChecker
    ... embed_dynamic = FeatureEmbedder(
    ...     cardinalities=[5, 5],
    ...     embedding_dims=[6, 9],
    ...     prefix='embed_dynamic_',
    ... )

    The above snippet with four :class:`nn.Embedding` corresponding to the one static and two dynamic categorical
    features. The `(input_dim, output_dim)` of these layers are going to be `(2, 3)`, `(5, 6)`, and `(5, 9)`.
    The created `assemble_feature` instance will not handle real-valued features.

    The subset of feature types to be used by the :class:`FeatureAssembler` instance is determined using corresponding
    constructor parameters. Here is an example that constructs a feature assembler consuming only real-valued features.

    >>> N, T = 50, 168
    >>> assemble_feature = FeatureAssembler(
    ...     T=T,
    ...     # use_static_cat=True,
    ...     # use_static_real=False,
    ...     # use_dynamic_cat=True,
    ...     # use_dynamic_real=False,
    ...     embed_static=embed_static,
    ...     embed_dynamic=embed_dynamic
    ... )

    When the `__call__`, `forward`, or `hybrid_forward` methods of a :class:`FeatureAssembler` are called, we always
    have to pass a full set of features. Missing features are represented as zero tensors with a suitable shape.

    For example,

    >>> import mxnet as mx
    >>> feat_static_cat = mx.nd.random.uniform(0, 2, shape=(N, 1)).floor()
    >>> feat_dynamic_cat = mx.nd.random.uniform(0, 5, shape=(N, 168, 2)).floor()
    >>> feat_static_real = mx.nd.zeros(shape=(N, 1,)) # empty feature
    >>> feat_dynamic_real = mx.nd.zeros(shape=(N, T, 1,)) # empty feature

    After initializing the embedder parameters to one and instantiating some random `static_cat` and
    `dynamic_cat` vectors,

    >>> assemble_feature.collect_params().initialize(mx.initializer.One())

    one can do a forward pass as follows.

    >>> assembled_feature = assemble_feature(feat_static_cat, feat_static_real, feat_dynamic_cat, feat_dynamic_real)
    >>> assembled_feature.shape
    (50, 168, 20)
    >>>

    However, relative order of `static_cat` and `dynamic_cat` in the call above is determined by the fact that
    `use_static_cat` is defined before `use_dynamic_cat` in the class constructor.
    """

    @validated()
    def __init__(
        self,
        T: int,
        use_static_cat: bool = False,
        use_static_real: bool = False,
        use_dynamic_cat: bool = False,
        use_dynamic_real: bool = False,
        embed_static: Optional[FeatureEmbedder] = None,
        embed_dynamic: Optional[FeatureEmbedder] = None,
        dtype: DType = np.float32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert T > 0, "The value of `T` should be > 0"

        self.T = T
        self.dtype = dtype
        self.use_static_cat = use_static_cat
        self.use_static_real = use_static_real
        self.use_dynamic_cat = use_dynamic_cat
        self.use_dynamic_real = use_dynamic_real
        self.embed_static: Callable[[Tensor], Tensor] = embed_static or (
            lambda x: x
        )
        self.embed_dynamic: Callable[[Tensor], Tensor] = embed_dynamic or (
            lambda x: x
        )

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        feat_dynamic_cat: Tensor,
        feat_dynamic_real: Tensor,
    ) -> Tensor:
        processed_features = [
            self.process_static_cat(F, feat_static_cat),
            self.process_static_real(F, feat_static_real),
            self.process_dynamic_cat(F, feat_dynamic_cat),
            self.process_dynamic_real(F, feat_dynamic_real),
        ]

        return F.concat(*processed_features, dim=-1)

    def process_static_cat(self, F, feature: Tensor) -> Tensor:
        feature = self.embed_static(feature.astype(self.dtype))
        return F.tile(feature.expand_dims(axis=1), reps=(1, self.T, 1))

    def process_dynamic_cat(self, F, feature: Tensor) -> Tensor:
        return self.embed_dynamic(feature.astype(self.dtype))

    def process_static_real(self, F, feature: Tensor) -> Tensor:
        return F.tile(feature.expand_dims(axis=1), reps=(1, self.T, 1))

    def process_dynamic_real(self, F, feature: Tensor) -> Tensor:
        return feature

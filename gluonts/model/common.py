# Standard library imports
import types
import typing

# Third-party imports
import mxnet as mx
import numpy as np

# Tensor type for HybridBlocks in Gluon
Tensor = typing.Union[mx.nd.NDArray, mx.sym.Symbol]

# Type of tensor-transforming functions in Gluon
TensorTransformer = typing.Callable[[types.ModuleType, Tensor], Tensor]

# untyped global configuration passed to model components
GlobalConfig = typing.Dict[str, typing.Any]

# to annotate Numpy parameter
NPArrayLike = typing.Union[int, float, np.ndarray]

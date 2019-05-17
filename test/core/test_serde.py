# Standard library imports
from pathlib import Path
from typing import List, NamedTuple

# Third-party imports
import mxnet as mx
import pytest
from pydantic import BaseModel

# First-party imports
from gluonts.core import serde


# Example Types
# -------------


class Span(NamedTuple):
    path: Path
    line: int


class BestEpochInfo(NamedTuple):
    params_path: Path
    epoch_no: int
    metric_value: float


class CategoricalFeatureInfo(BaseModel):
    name: str
    cardinality: int


class MyGluonBlock(mx.gluon.HybridBlock):
    def __init__(
        self,
        feature_infos: List[CategoricalFeatureInfo],
        feature_dims: List[int],
    ) -> None:
        super().__init__()
        self.feature_infos = feature_infos
        self.feature_dims = feature_dims

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError

    # required for all user-defined types
    def __getnewargs_ex__(self):
        return (self.feature_infos, self.feature_dims), dict()

    def __eq__(self, that) -> bool:
        if isinstance(that, MyGluonBlock):
            return self.__getnewargs_ex__() == that.__getnewargs_ex__()
        else:
            return False


# Example Instances
# -----------------

best_epoch_info = BestEpochInfo(
    params_path=Path('foo/bar'), epoch_no=1, metric_value=0.5
)

feature_info = CategoricalFeatureInfo(name='cat', cardinality=10)

custom_type = MyGluonBlock(feature_infos=[feature_info], feature_dims=[10])

list_container = [best_epoch_info, feature_info, custom_type, 42, 0.7, 'fx']

dict_container = dict(
    best_epoch_info=best_epoch_info,
    feature_info=feature_info,
    custom_type=custom_type,
)

simple_types = [1, 42.0, 'Oh, Romeo']  # float('nan')

complex_types = [Path('foo/bar'), best_epoch_info, feature_info, custom_type]

container_types = [list_container, dict_container]

examples = simple_types + complex_types + container_types  # type: ignore


@pytest.mark.parametrize('e', examples)
def test_binary_serialization(e) -> None:
    assert e == serde.load_binary(serde.dump_binary(e))


@pytest.mark.parametrize('e', examples)
def test_json_serialization(e) -> None:
    assert e == serde.load_json(serde.dump_json(e))


@pytest.mark.parametrize('e', examples)
def test_code_serialization(e) -> None:
    assert e == serde.load_code(serde.dump_code(e))

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
import functools
import inspect
import logging
import os
import re
from collections import OrderedDict
from functools import singledispatch
from pydoc import locate
from typing import Any, Type, TypeVar, Union

# Third-party imports
import mxnet as mx
from pydantic import BaseConfig, BaseModel, ValidationError, create_model

# First-party imports
from gluonts.core.exception import GluonTSHyperparametersError
from gluonts.core.serde import dump_code
from gluonts.monkey_patch import monkey_patch_property_metaclass  # noqa: F401

# Relative imports
from ._base import fqname_for

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

A = TypeVar('A')


def from_hyperparameters(cls: Type[A], **hyperparameters) -> A:
    Model = getattr(cls.__init__, 'Model', None)

    if not Model:
        raise AttributeError(
            f'Cannot find attribute Model attached to the '
            f'{fqname_for(cls)}. Most probably you have forgotten to mark '
            f'the class constructor as @validated().'
        )

    try:
        return cls(**Model(**hyperparameters).__values__)  # type: ignore
    except ValidationError as e:
        raise GluonTSHyperparametersError from e


@singledispatch
def equals(this: Any, that: Any) -> bool:
    if type(this) != type(that):
        return False
    elif hasattr(this, '__init_args__') and hasattr(that, '__init_args__'):
        params1 = getattr(this, '__init_args__')
        params2 = getattr(that, '__init_args__')

        pnames1 = params1.keys()
        pnames2 = params2.keys()

        if not pnames1 == pnames2:
            return False

        for name in pnames1:
            x = params1[name]
            y = params2[name]
            if not equals(x, y):
                return False

        return True
    else:
        return this == that


@equals.register(mx.gluon.HybridBlock)
def equals_representable_block(
    this: mx.gluon.HybridBlock, that: mx.gluon.HybridBlock
) -> bool:
    if not equals(this, that):
        return False

    if not equals_parameter_dict(this.collect_params(), that.collect_params()):
        return False

    return True


@equals.register(mx.gluon.ParameterDict)
def equals_parameter_dict(
    this: mx.gluon.ParameterDict, that: mx.gluon.ParameterDict
) -> bool:
    if type(this) != type(that):
        return False

    this_prefix_length = len(this.prefix)
    that_prefix_length = len(that.prefix)

    this_param_names_stripped = {
        key[this_prefix_length:] if key.startswith(this.prefix) else key
        for key in this.keys()
    }
    that_param_names_stripped = {
        key[that_prefix_length:] if key.startswith(that.prefix) else key
        for key in that.keys()
    }

    if not this_param_names_stripped == that_param_names_stripped:
        return False

    for this_param_name, that_param_name in zip(this.keys(), that.keys()):
        x = this[this_param_name].data().asnumpy()
        y = that[that_param_name].data().asnumpy()
        if not mx.test_utils.almost_equal(x, y, equal_nan=True):
            return False

    return True


class ConfigBase(BaseModel):
    """Base config for components with @validated constructors."""

    class Config(BaseConfig):
        arbitrary_types_allowed = True


def validated(base_model=None):
    """
    Decorates a constructor with typed arguments with validation logic which
    delegates to a Pydantic model. If `base_model` is not provided, an implicit
    model is synthesized. If `base_model` is provided, its fields and types
    should be consistent with the constructor arguments and the model should
    extend `ConfigBase`.
    """

    def validator(ctor):
        ctor_clsnme = dict(inspect.getmembers(ctor))['__qualname__'].split(
            '.'
        )[0]
        ctor_params = inspect.signature(ctor).parameters
        ctor_fields = {
            param.name: (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else Any,
                param.default
                if param.default != inspect.Parameter.empty
                else ...,
            )
            for param in ctor_params.values()
            if param.name != 'self'
            and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        }

        if base_model is None:
            CtorModel = create_model(
                f'{ctor_clsnme}Model',
                __config__=ConfigBase.Config,
                **ctor_fields,
            )
        else:
            CtorModel = create_model(
                f'{ctor_clsnme}Model', __base__=base_model, **ctor_fields
            )

        @functools.wraps(ctor)
        def ctor_wrapper(*args, **kwargs):
            self, *args = args

            nmargs = {
                name: arg
                for (name, param), arg in zip(
                    list(ctor_params.items()), [self] + args
                )
                if name != 'self'
            }
            model = CtorModel(**{**nmargs, **kwargs})

            # merge nmargs, kwargs, and the model fields into a single dict
            all_args = {**nmargs, **kwargs, **model.__values__}

            # save the merged dictionary for Representable use, but only of the
            # __init_args__ is not already set in order to avoid overriding a
            # value set by a subclass constructor in super().__init__ calls
            if not getattr(self, '__init_args__', {}):
                self.__init_args__ = OrderedDict(
                    {
                        name: arg
                        for name, arg in sorted(all_args.items())
                        if type(arg) != mx.gluon.ParameterDict
                    }
                )
                self.__class__.__getnewargs_ex__ = validated_getnewargs_ex
                self.__class__.__repr__ = validated_repr

            return ctor(self, **all_args)

        # attach the model as the attribute of the constructor wrapper
        setattr(ctor_wrapper, 'Model', CtorModel)

        return ctor_wrapper

    return validator


def validated_repr(self) -> str:
    return dump_code(self)


def validated_getnewargs_ex(self):
    return (), self.__init_args__


class MXContext:
    @classmethod
    def validate(cls, v: Union[str, mx.Context]) -> mx.Context:
        if isinstance(v, mx.Context):
            return v

        m = re.search(
            r'^(?P<device_type>cpu|gpu)(\((?P<device_id>\d+)\))?$', v
        )

        if m:
            return mx.Context(m['device_type'], int(m['device_id'] or 0))
        else:
            raise ValueError(
                f'bad MXNet context {v}, expected an '
                f'mx.context.Context its string representation'
            )

    @classmethod
    def get_validators(cls) -> mx.Context:
        yield cls.validate


mx.Context.validate = MXContext.validate
mx.Context.get_validators = MXContext.get_validators


def has_gpu_support():
    try:
        mx.nd.array([1, 2, 3], ctx=mx.gpu(0))
        return True
    except mx.MXNetError:
        return False


def check_gpu_support():
    logger.info(f'MXNet GPU support is {"ON" if has_gpu_support() else "OFF"}')


class DType:
    @classmethod
    def get_validators(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, str):
            return locate(v)
        if isinstance(v, type):
            return v
        else:
            raise ValueError(
                f'bad value {v} of type {type(v)}, expected a type or a string'
            )

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
from . import fqname_for

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

A = TypeVar('A')


def from_hyperparameters(cls: Type[A], **hyperparameters) -> A:
    """
    Reflectively create an instance of a class with a :func:`validated`
    initializer.

    Parameters
    ----------
    cls
        The type ``A`` of the component to be instantiated.
    hyperparameters
        A dictionary of key-value pairs to be used as parameters to the
        component initializer.

    Returns
    -------
    A
        An instance of the given class.

    Raises
    ------
    GluonTSHyperparametersError
        Wraps a :class:`ValidationError` thrown when validating the
        initializer parameters.
    """
    Model = getattr(cls.__init__, 'Model', None)

    if not Model:
        raise AttributeError(
            f'Cannot find attribute Model attached to the '
            f'{fqname_for(cls)}. Most probably you have forgotten to mark '
            f'the class initializer as @validated().'
        )

    try:
        return cls(**Model(**hyperparameters).__values__)  # type: ignore
    except ValidationError as e:
        raise GluonTSHyperparametersError from e


@singledispatch
def equals(this: Any, that: Any) -> bool:
    """
    Structural equality check between two objects of arbitrary type.

    Two objects ``this`` and ``that`` are defined to be structurally equal
    if and only if the following criteria are satisfied:

    1. Their types match.
    2. If their initializer are :func:`validated`, their initializer arguments
       are pairlise structurally equal.
    3. If their initializer are not :func:`validated`, they are referentially
       equal (i.e. ``this == that``).

    In addition, the function dispatches to specialized implementations based
    on the type of the first argument, so the above conditions might be
    sticter for certain types.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals_representable_block
        Specialization for Gluon :class:`~mxnet.gluon.HybridBlock` input
        arguments.
    equals_parameter_dict
        Specialization for Gluon :class:`~mxnet.gluon.ParameterDict` input
        arguments.
    """
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
    """
    Structural equality check between two :class:`~mxnet.gluon.HybridBlock`
    objects with :func:`validated` initializers.

    Two blocks ``this`` and ``that`` are considered *structurally equal* if all
    the conditions of :func:`equals` are met, and in addition their parameter
    dictionaries obtained with
    :func:`~mxnet.gluon.block.Block.collect_params` are also structurally
    equal.

    Specializes :func:`equals` for invocations where the first parameter is an
    instance of the :class:`~mxnet.gluon.HybridBlock` class.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals
        Dispatching function.
    equals_parameter_dict
        Specialization of :func:`equals` for Gluon
        :class:`~mxnet.gluon.ParameterDict` input arguments.
    """
    if not equals(this, that):
        return False

    if not equals_parameter_dict(this.collect_params(), that.collect_params()):
        return False

    return True


@equals.register(mx.gluon.ParameterDict)
def equals_parameter_dict(
    this: mx.gluon.ParameterDict, that: mx.gluon.ParameterDict
) -> bool:
    """
    Structural equality check between two :class:`~mxnet.gluon.ParameterDict`
    objects.

    Two parameter dictionaries ``this`` and ``that`` are considered
    *structurally equal* if the following conditions are satisfied:

    1. They contain the same keys (modulo the key prefix which is stripped).
    2. The data in the corresponding value pairs is equal, as defined by the
       :func:`~mxnet.test_utils.almost_equal` function (in this case we call
       the function with ``equal_nan=True``, that is, two aligned ``NaN``
       values are always considered equal).

    Specializes :func:`equals` for invocations where the first parameter is an
    instance of the :class:`~mxnet.gluon.ParameterDict` class.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals
        Dispatching function.
    """
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


class BaseValidatedInitializerModel(BaseModel):
    """
    Base Pydantic model for components with :func:`validated` initializers.

    See Also
    --------
    validated
        Decorates an initializer methods with argument validation logic.
    """

    class Config(BaseConfig):
        """
        `Config <https://pydantic-docs.helpmanual.io/#model-config>`_ for the
        Pydantic model inherited by all :func:`validated` initializers.

        Allows the use of arbitrary type annotations in initializer parameters.
        """

        arbitrary_types_allowed = True


def validated(base_model=None):
    """
    Decorates an ``__init__`` method with typed parameters with validation
    and auto-conversion logic.

    >>> class ComplexNumber:
    ...     @validated()
    ...     def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
    ...         self.x = x
    ...         self.y = y

    Classes with decorated initializers can be instantiated using arguments of
    another type (e.g. an ``y`` argument of type ``str`` ). The decorator
    handles the type conversion logic.

    >>> c = ComplexNumber(y='42')
    >>> (c.x, c.y)
    (0.0, 42.0)

    If the bound argument cannot be converted, the decorator throws an error.

    >>> c = ComplexNumber(y=None)
    Traceback (most recent call last):
        ...
    pydantic.error_wrappers.ValidationError: 1 validation error
    y
      value is not a valid float (type=type_error.float)

    Internally, the decorator delegates all validation and conversion logic to
    `a Pydantic model <https://pydantic-docs.helpmanual.io/>`_, which can be
    accessed through the ``Model`` attribute of the decorated initiazlier.

    >>> ComplexNumber.__init__.Model
    <class 'ComplexNumberModel'>

    The Pydantic model is synthesized automatically from on the parameter
    names and types of the decorated initializer. In the ``ComplexNumber``
    example, the synthesized Pydantic model corresponds to the following
    definition.

    >>> class ComplexNumberModel(BaseValidatedInitializerModel):
    ...     x: float = 0.0
    ...     y: float = 0.0


    Clients can optionally customize the base class of the synthesized
    Pydantic model using the ``base_model`` decorator parameter. The default
    behavior uses :class:`BaseValidatedInitializerModel` and its
    `model config <https://pydantic-docs.helpmanual.io/#config>`_.

    See Also
    --------
    BaseValidatedInitializerModel
        Default base class for all synthesized Pydantic models.
    """

    def validator(init):
        init_qualname = dict(inspect.getmembers(init))['__qualname__']
        init_clsnme = init_qualname.split('.')[0]
        init_params = inspect.signature(init).parameters
        init_fields = {
            param.name: (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else Any,
                param.default
                if param.default != inspect.Parameter.empty
                else ...,
            )
            for param in init_params.values()
            if param.name != 'self'
            and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        }

        if base_model is None:
            PydanticModel = create_model(
                model_name=f'{init_clsnme}Model',
                __config__=BaseValidatedInitializerModel.Config,
                **init_fields,
            )
        else:
            PydanticModel = create_model(
                model_name=f'{init_clsnme}Model',
                __base__=base_model,
                **init_fields,
            )

        def validated_repr(self) -> str:
            return dump_code(self)

        def validated_getnewargs_ex(self):
            return (), self.__init_args__

        @functools.wraps(init)
        def init_wrapper(*args, **kwargs):
            self, *args = args

            nmargs = {
                name: arg
                for (name, param), arg in zip(
                    list(init_params.items()), [self] + args
                )
                if name != 'self'
            }
            model = PydanticModel(**{**nmargs, **kwargs})

            # merge nmargs, kwargs, and the model fields into a single dict
            all_args = {**nmargs, **kwargs, **model.__values__}

            # save the merged dictionary for Representable use, but only of the
            # __init_args__ is not already set in order to avoid overriding a
            # value set by a subclass initializer in super().__init__ calls
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

            return init(self, **all_args)

        # attach the Pydantic model as the attribute of the initializer wrapper
        setattr(init_wrapper, 'Model', PydanticModel)

        return init_wrapper

    return validator


class MXContext:
    """
    Defines `custom data type validation
    <https://pydantic-docs.helpmanual.io/#custom-data-types>`_ for
    the :class:`~mxnet.context.Context` data type.
    """

    @classmethod
    def validate(cls, v: Union[str, mx.Context]) -> mx.Context:
        if isinstance(v, mx.Context):
            return v

        m = re.search(r'^(?P<dev_type>cpu|gpu)(\((?P<dev_id>\d+)\))?$', v)

        if m:
            return mx.Context(m['dev_type'], int(m['dev_id'] or 0))
        else:
            raise ValueError(
                f'bad MXNet context {v}, expected either an '
                f'mx.context.Context or its string representation'
            )

    @classmethod
    def __get_validators__(cls) -> mx.Context:
        yield cls.validate


mx.Context.validate = MXContext.validate
mx.Context.__get_validators__ = MXContext.__get_validators__


def has_gpu_support() -> bool:
    """
    Checks if the currently installed MXNet version has GPU support.
    """
    try:
        mx.nd.array([1, 2, 3], ctx=mx.gpu(0))
        return True
    except mx.MXNetError:
        return False


def check_gpu_support() -> None:
    """
    Emits a log line that indicates whether the currently installed MXNet
    version has GPU support.
    """
    logger.info(f'MXNet GPU support is {"ON" if has_gpu_support() else "OFF"}')


class DType:
    """
    Defines `custom data type validation
    <https://pydantic-docs.helpmanual.io/#custom-data-types>`_ for ``type``
    instances.

    Parameters annotated with :class:`DType` can be bound to string arguments
    representing the fully-qualified type name. The validation logic
    defined here attempts to automatically load the type as part of the
    conversion process.
    """

    @classmethod
    def __get_validators__(cls):
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

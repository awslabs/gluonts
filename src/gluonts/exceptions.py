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

from typing import Any

from pydantic.error_wrappers import ValidationError, display_errors


class GluonTSException(Exception):
    """
    Base class for all GluonTS exceptions.
    """

    @classmethod
    def guard(cls, condition, *args, **kwargs):
        if not condition:
            raise cls(*args, **kwargs)


class GluonTSHyperparametersError(GluonTSException, ValueError):
    """
    An exception wrapping a Pydantic ``ValidationError``, usually thrown when
    the validation of a :func:`~gluonts.core.component.validated` initializer
    fails.
    """

    __cause__: ValidationError

    def __str__(self, *args, **kwargs):
        return (
            "The following errors occurred when trying to "
            "validate the algorithm hyperparameters:\n"
            f"{display_errors(self.__cause__.errors())}"
        )


class GluonTSDataError(GluonTSException):
    """
    An exception indicating an error with the input data.
    """

    pass


class GluonTSUserError(GluonTSException):
    """
    An exception indicating a user error.
    """

    pass


class GluonTSDateBoundsError(GluonTSException):
    """
    An exception indicating that .
    """

    pass


def assert_gluonts(
    exception_class: type, condition: Any, message: str, *args, **kwargs
) -> None:
    """
    If the given ``condition`` is ``False``, raises an exception of type
    ``exception_class`` with a message formatted from the ``message`` pattern
    using the ``args`` and ``kwargs`` strings.

    Parameters
    ----------
    exception_class
        The exception class of the raised exception.
    condition
        The condition that must be violated in order to raise the exception.
    message
        A message to pass as the only argument to the exception initializer.
    args
        An optional list of positional arguments to use when formatting the
        exception message.
    kwargs
        An optional list of key-value arguments to use when formatting the
        exception message.
    """
    if not condition:
        raise exception_class(message.format(*args, **kwargs))


def assert_data_error(condition: Any, message: str, *args, **kwargs) -> None:
    """
    Delegates to :func:`assert_gluonts` with a fixed ``exception_class`` value
    of ``GluonTSDataError``.

    Parameters
    ----------
    condition
        The condition that must be violated in order to raise the exception.
    message
        A message to pass as the only argument to the exception initializer.
    args
        An optional list of positional arguments to use when formatting the
        exception message.
    kwargs
        An optional list of key-value arguments to use when formatting the
        exception message.
    """
    assert_gluonts(GluonTSDataError, condition, message, *args, **kwargs)

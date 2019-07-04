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
from typing import Callable

# Third-party imports
from pydantic.error_wrappers import ValidationError, display_errors


class GluonTSException(Exception):
    """
    Base class for all GluonTS exceptions.
    """

    pass


class GluonTSFatalError(GluonTSException):
    """
    An exception indicating an arbitrary cause that prohibits further
    execution of the program.
    """

    pass


class GluonTSForecasterNotFoundError(GluonTSException):
    """
    An exception indicating that a forecaster identified by the given
    name cannot be found in the current environment.
    """

    pass


class GluonTSHyperparameterParseError(GluonTSException):
    """
    An exception indicating a parse error when trying to re-interpret a
    string value ``value`` for a parameter ``key`` as a value of type ``type``.
    """

    __cause__: ValueError

    def __init__(self, key, value, type):
        self.key = key
        self.value = value
        self.type = type

    def __str__(self, *args, **kwargs):
        return (
            f'Error when trying to re-interpret string value "{self.value}" '
            f'for parameter {self.key} as a {self.type}:\n'
            f'{repr(self.__cause__)}'
        )


class GluonTSHyperparametersError(GluonTSException):
    """
    An exception wrapping a Pydantic ``ValidationError``, usually thrown when
    the validation of a :func:`~gluonts.core.component.validated` initializer
    fails.
    """

    __cause__: ValidationError

    def __str__(self, *args, **kwargs):
        return (
            f'The following errors occurred when trying to '
            f'validate the algorithm hyperparameters:\n'
            f'{display_errors(self.__cause__.errors())}'
        )


class GluonTSDataError(GluonTSException):
    """
    An exception indicating an error with the input data.
    """

    pass


class GluonTSInvalidRequestException(GluonTSException):
    """
    An exception indicating an invalid inference request.
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
    exception_class: type, condition: bool, message: str, *args, **kwargs
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


def assert_data_error(condition: bool, message: str, *args, **kwargs) -> None:
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


def reraise_error(
    origin_class: type,
    message: str = None,
    target_class: type = GluonTSUserError,
) -> Callable:
    """
    Decorator that converts `Origin` to `Target` exception, where `Origin` is
    not an instance of `CustomerError`.

    If `message` is not provided, the message of the causing exception is
    simply past to the ``GluonTSUserError``. If `message` is specified, the
    ``GluonTSUserError`` will be constructed with that message and the causing
    exception is added as the cause.

    Parameters
    ----------
    origin_class
        The type of the original exception.
    message
        A message to pass to the re-raised exception.
    target_class
        The type of hte re-raised exception.

    Returns
    -------

    """

    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except origin_class as error:
                import traceback

                traceback.print_exc()
                error_message = message or getattr(error, 'message', None)
                if error_message is None:
                    raise target_class(message=error)
                else:
                    raise target_class(message=error_message, caused_by=error)

        return inner

    return decorator

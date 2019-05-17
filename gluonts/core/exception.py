# Standard library imports
import functools

# Third-party imports
from pydantic.error_wrappers import ValidationError, display_errors


class GluonTSException(Exception):
    pass


class GluonTSFatalError(GluonTSException):
    pass


class GluonTSHyperparameterParseError(Exception):
    __cause__: ValueError

    def __init__(self, key, value, type):
        self.key = key
        self.value = value
        self.type = type

    def __str__(self, *args, **kwargs):
        return (
            f'Error when trying to re-interpret '
            f'string value "{self.value}" for parameter {self.key} as a {self.type}:\n'
            f'{repr(self.__cause__)}'
        )


class GluonTSHyperparametersError(GluonTSException):
    __cause__: ValidationError

    def __str__(self, *args, **kwargs):
        return (
            f'The following errors occurred when trying to '
            f'validate the algorithm hyperparameters:\n'
            f'{display_errors(self.__cause__.errors())}'
        )


class GluonTSDataError(GluonTSException):
    pass


class GluonTSInvalidRequestException(GluonTSException):
    pass


class GluonTSUserError(GluonTSException):
    pass


class GluonTSDateBoundsError(GluonTSException):
    pass


def assert_gluonts(exception_class, condition, message, *args, **kwargs):
    if not condition:
        raise exception_class(message.format(*args, **kwargs))


def assert_data_error(condition, message, *args, **kwargs):
    if not condition:
        raise GluonTSDataError(message.format(*args, **kwargs))


def reraise_error(Origin, message=None, Target=GluonTSUserError):
    '''
    Decorator that converts `Origin` to `Target` exception, where `Origin` is not
    an instance of `CustomerError`.

    If `message` is not provided, the message of the causing exception is simply past to
    the CustomerError. If `message` is specified, the CustomerError will be constructed
    with that message and the causing exception is added as the cause.

    :param origin:
    :param target:
    :return:
    '''

    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Origin as error:
                import traceback

                traceback.print_exc()
                error_message = message or getattr(error, 'message', None)
                if error_message is None:
                    raise Target(message=error)
                else:
                    raise Target(message=error_message, caused_by=error)

        return inner

    return decorator

# Third-party imports
import mxnet as mx


def __my__setattr__(cls, key, value):
    obj = cls.__dict__.get(key)
    if obj and isinstance(obj, mx.base._MXClassPropertyDescriptor):
        return obj.__set__(cls, value)

    return super(mx.base._MXClassPropertyMetaClass, cls).__setattr__(
        key, value
    )


mx.base._MXClassPropertyMetaClass.__setattr__ = __my__setattr__

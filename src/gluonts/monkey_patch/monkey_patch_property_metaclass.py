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

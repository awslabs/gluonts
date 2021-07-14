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


def _orjson():
    from functools import partial
    import orjson

    dumps = partial(orjson.dumps, option=orjson.OPT_SERIALIZE_NUMPY)

    def load(fp):
        return orjson.loads(fp.read())

    def dump(obj, fp):
        return print(dumps(obj), file=fp)

    return "orjson", {
        "loads": orjson.loads,
        "load": load,
        "dumps": dumps,
        "dump": dump,
    }


def _ujson():
    import ujson

    return "ujson", vars(ujson)


def _json():
    import json
    import warnings

    warnings.warn(
        "Using `json`-module for json-handling. "
        "Consider installing one of `orjson`, `ujson` "
        "to speed up serialization and deserialization."
    )

    return "json", vars(json)


for fn in _orjson, _ujson, _json:
    try:
        variant, _methods = fn()

        load = _methods["load"]
        loads = _methods["loads"]
        dump = _methods["dump"]
        dumps = _methods["dumps"]
        break
    except ImportError:
        continue

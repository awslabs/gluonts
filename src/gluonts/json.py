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

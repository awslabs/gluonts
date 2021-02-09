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

import ast
from functools import singledispatch


@singledispatch
def parse_expr(v):
    raise ValueError(v)


@parse_expr.register(ast.Num)
def parse_num(v: ast.Num):
    return v.n


@parse_expr.register(ast.UnaryOp)
def parse_unary_op(v: ast.UnaryOp):
    operand = parse_expr(v.operand)

    assert isinstance(operand, (float, int))
    assert isinstance(v.op, ast.USub)

    return -operand


@parse_expr.register(ast.Str)
def parse_str(v: ast.Str):
    return v.s


@parse_expr.register(ast.List)
def parse_list(v: ast.List):
    return list(map(parse_expr, v.elts))


@parse_expr.register(ast.Tuple)
def parse_tuple(v: ast.Tuple):
    return tuple(map(parse_expr, v.elts))


@parse_expr.register(ast.Dict)
def parse_dict(v: ast.Dict):
    keys = map(parse_expr, v.keys)
    values = map(parse_expr, v.values)
    return dict(zip(keys, values))


@parse_expr.register(ast.Set)
def parse_set(v: ast.Set):
    return set(map(parse_expr, v.elts))


@parse_expr.register(ast.keyword)
def parse_keyword(v: ast.keyword):
    return v.arg, parse_expr(v.value)


@parse_expr.register(ast.NameConstant)
def parse_name_constant(v: ast.NameConstant):
    return v.value


@parse_expr.register(ast.Constant)
def parse_constant(v: ast.Constant):
    return v.value


@parse_expr.register(ast.Attribute)
def parse_attribute(v: ast.Attribute, path=()):
    if isinstance(v.value, ast.Name):
        return {
            "__kind__": "type",
            "class": ".".join((v.value.id, v.attr) + path),
        }
    else:
        assert isinstance(v.value, ast.Attribute)
        return parse_attribute(v.value, (v.attr,) + path)


@parse_expr.register(ast.Name)
def parse_name(v: ast.Name):
    return {"__kind__": "type", "class": v.id}


@parse_expr.register(ast.Call)
def parse_expr_call(v: ast.Call):
    args = list(map(parse_expr, v.args))
    kwargs = dict(map(parse_keyword, v.keywords))

    class_name = parse_expr(v.func)["class"]
    obj = {"__kind__": "instance", "class": class_name}

    if args:
        obj["args"] = args

    if kwargs:
        obj["kwargs"] = kwargs

    return obj


def parse(s):
    module = ast.parse(s)
    expr = module.body[0]
    return parse_expr(expr.value)

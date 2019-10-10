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

import numpy as np


def jsonify_floats(json_object):
    """
    Traverses through the JSON object and converts non JSON-spec compliant
    floats(nan, -inf, inf) to their string representations.

    Parameters
    ----------
    json_object
        JSON object
    """
    if isinstance(json_object, dict):
        return {k: jsonify_floats(v) for k, v in json_object.items()}
    elif isinstance(json_object, list):
        return [jsonify_floats(item) for item in json_object]
    elif isinstance(json_object, float):
        if np.isnan(json_object):
            return "NaN"
        elif np.isposinf(json_object):
            return "Infinity"
        elif np.isneginf(json_object):
            return "-Infinity"
        return json_object
    return json_object

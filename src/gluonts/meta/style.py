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

from pygments.style import Style
from pygments.token import Comment, Keyword, Name, String, Number, Operator

from . import colors


class Dark(Style):
    # transparent
    background_color = "rgba(0, 0, 0, 0.0)"

    styles = {
        Comment: "italic #888",
        Comment.Preproc: "noitalic #888",
        Keyword: f"bold {colors.SALMON}",
        Keyword.Pseudo: "nobold",
        Keyword.Type: f"nobold {colors.SALMON}",
        Operator: f"bold {colors.SALMON}",
        Operator.Word: f"bold {colors.SALMON}",
        Name.Builtin: colors.YELLOW,
        Name.Function: f"{colors.GREEN}",
        Name.Class: f"bold {colors.GREEN}",
        Name.Namespace: f"bold {colors.GREEN}",
        Name.Variable: colors.YELLOW,
        Name.Constant: colors.RED,
        Name.Label: colors.YELLOW,
        Name.Attribute: colors.YELLOW,
        Name.Tag: f"bold {colors.SALMON}",
        Name.Decorator: colors.SALMON,
        String: colors.SALMON,
        String.Doc: "italic",
        Number: colors.GREEN,
    }

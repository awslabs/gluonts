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

from toolz import curry


from gluonts.itertools import columns_to_rows


@curry
def tag(name, data):
    return f"<{name}>{data}</{name}>"


def html_table(columns):
    head = " ".join(map(tag("th"), columns))

    rows = "\n".join(
        map(
            tag("tr"),
            [
                "".join(map(tag("td"), row.values()))
                for row in columns_to_rows(columns)
            ],
        )
    )

    return f"""
        <table>
            <thead>
                {head}
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>

        """

    # {len(self)} rows × {len(self.columns)} columns

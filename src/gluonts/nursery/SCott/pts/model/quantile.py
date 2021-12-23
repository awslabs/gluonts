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


import re
from typing import NamedTuple, Union


class Quantile(NamedTuple):
    value: float
    name: str

    @property
    def loss_name(self):
        return f"QuantileLoss[{self.name}]"

    @property
    def weighted_loss_name(self):
        return f"wQuantileLoss[{self.name}]"

    @property
    def coverage_name(self):
        return f"Coverage[{self.name}]"

    @classmethod
    def checked(cls, value: float, name: str) -> "Quantile":
        if not 0 <= value <= 1:
            raise Exception(f"quantile value should be in [0, 1] but found {value}")

        return Quantile(value, name)

    @classmethod
    def from_float(cls, quantile: float) -> "Quantile":
        assert isinstance(quantile, float)
        return cls.checked(value=quantile, name=str(quantile))

    @classmethod
    def from_str(cls, quantile: str) -> "Quantile":
        assert isinstance(quantile, str)
        try:
            return cls.checked(value=float(quantile), name=quantile)
        except ValueError:
            m = re.match(r"^p(\d{2})$", quantile)

            if m is None:
                raise Exception(
                    "Quantile string should be of the form "
                    f'"p10", "p50", ... or "0.1", "0.5", ... but found {quantile}'
                )
            else:
                quantile_float: float = int(m.group(1)) / 100
                return cls(value=quantile_float, name=str(quantile_float))

    @classmethod
    def parse(cls, quantile: Union["Quantile", float, str]) -> "Quantile":
        """Produces equivalent float and string representation of a given
        quantile level.

        >>> Quantile.parse(0.1)
        Quantile(value=0.1, name='0.1')

        >>> Quantile.parse('0.2')
        Quantile(value=0.2, name='0.2')

        >>> Quantile.parse('0.20')
        Quantile(value=0.2, name='0.20')

        >>> Quantile.parse('p99')
        Quantile(value=0.99, name='0.99')

        Parameters
        ----------
        quantile
            Quantile, can be a float a str representing a float e.g. '0.1' or a
            quantile string of the form 'p0.1'.

        Returns
        -------
        Quantile
            A tuple containing both a float and a string representation of the
            input quantile level.
        """
        if isinstance(quantile, Quantile):
            return quantile
        elif isinstance(quantile, float):
            return cls.from_float(quantile)
        else:
            return cls.from_str(quantile)

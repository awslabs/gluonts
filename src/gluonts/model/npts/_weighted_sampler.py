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
import numpy as np


class WeightedSampler:
    """
    Utility class for sampling indices based on unnormalized weights.

    """

    @staticmethod
    def sample(weights, num_samples):
        """
        Sample indices according to `weights`:
            `ix` is chosen with probability `weights`[`ix`]

        `weights` need not sum to 1.

        :param weights:
        :param num_samples:
        :return:
        """
        assert all(weights >= 0.0), "Sampling weights must be non-negative"
        # In the special case where all the weights are zeros, we want to
        # sample all indices uniformly
        weights = np.ones_like(weights) if sum(weights) == 0.0 else weights

        cumsum_weights = np.cumsum(weights)

        # Just for better readability.
        total_weight = cumsum_weights[-1]

        # Samples from the Uniform distribution: U(0, `total_weight`)
        uniform_samples = total_weight * np.random.random(num_samples)

        # Search for the last `ix` for each sample u ~ U(0, total weight)
        # such that u <= `cumsum`[`ix`]
        # This means `ix` is chosen with probability
        # `cumsum`[`ix`] - `cumsum`[`ix` - 1] = weights[ix]
        samples_ix = np.searchsorted(
            cumsum_weights, uniform_samples, side="left"
        )

        return samples_ix

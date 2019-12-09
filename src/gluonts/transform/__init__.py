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


__all__ = [
    "AddAgeFeature",
    "AddConstFeature",
    "AddObservedValuesIndicator",
    "AddTimeFeatures",
    "AdhocTransform",
    "AsNumpyArray",
    "BucketInstanceSampler",
    "CanonicalInstanceSplitter",
    "cdf_to_gaussian_forward_transform",
    "CDFtoGaussianTransform",
    "ConcatFeatures",
    "ExpandDimArray",
    "ExpectedNumInstanceSampler",
    "FilterTransformation",
    "FlatMapTransformation",
    "Identity",
    "InstanceSampler",
    "InstanceSplitter",
    "ListFeatures",
    "MapTransformation",
    "RemoveFields",
    "RenameFields",
    "SampleTargetDim",
    "SelectFields",
    "SetField",
    "SetFieldIfNotPresent",
    "shift_timestamp",
    "SimpleTransformation",
    "SwapAxes",
    "target_transformation_length",
    "TargetDimIndicator",
    "TestSplitSampler",
    "Transformation",
    "UniformSplitSampler",
    "VstackFeatures",
]

from ._base import (
    Transformation,
    Chain,
    Identity,
    MapTransformation,
    SimpleTransformation,
    AdhocTransform,
    FlatMapTransformation,
    FilterTransformation,
)

from .convert import (
    AsNumpyArray,
    ExpandDimArray,
    VstackFeatures,
    ConcatFeatures,
    SwapAxes,
    ListFeatures,
    TargetDimIndicator,
    SampleTargetDim,
    CDFtoGaussianTransform,
    cdf_to_gaussian_forward_transform,
)

from .dataset import TransformedDataset

from .feature import (
    target_transformation_length,
    AddObservedValuesIndicator,
    AddConstFeature,
    AddTimeFeatures,
    AddAgeFeature,
)

from .field import (
    RemoveFields,
    RenameFields,
    SetField,
    SetFieldIfNotPresent,
    SelectFields,
)


from .sampler import (
    InstanceSampler,
    UniformSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    BucketInstanceSampler,
    ContinuousTimePointSampler,
    ContinuousTimeUniformSampler,
)

from .split import (
    shift_timestamp,
    InstanceSplitter,
    CanonicalInstanceSplitter,
    ContinuousTimeInstanceSplitter,
)


# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)

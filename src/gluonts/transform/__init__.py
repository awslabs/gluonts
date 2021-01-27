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
    "AddAggregateLags",
    "AddConstFeature",
    "AddObservedValuesIndicator",
    "AddTimeFeatures",
    "AdhocTransform",
    "AsNumpyArray",
    "BucketInstanceSampler",
    "CanonicalInstanceSplitter",
    "cdf_to_gaussian_forward_transform",
    "CDFtoGaussianTransform",
    "Chain",
    "ConcatFeatures",
    "ContinuousTimeInstanceSplitter",
    "ContinuousTimePointSampler",
    "ContinuousTimeUniformSampler",
    "ContinuousTimePredictionSampler",
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
    "TransformedDataset",
    "TestSplitSampler",
    "ValidationSplitSampler",
    "Transformation",
    "UniformSplitSampler",
    "VstackFeatures",
    "MissingValueImputation",
    "LeavesMissingValues",
    "DummyValueImputation",
    "MeanValueImputation",
    "LastValueImputation",
    "CausalMeanValueImputation",
    "RollingMeanValueImputation",
]

from ._base import (
    AdhocTransform,
    Chain,
    FilterTransformation,
    FlatMapTransformation,
    Identity,
    MapTransformation,
    SimpleTransformation,
    Transformation,
)
from .convert import (
    AsNumpyArray,
    CDFtoGaussianTransform,
    ConcatFeatures,
    ExpandDimArray,
    ListFeatures,
    SampleTargetDim,
    SwapAxes,
    TargetDimIndicator,
    VstackFeatures,
    cdf_to_gaussian_forward_transform,
)
from .dataset import TransformedDataset
from .feature import (
    AddAgeFeature,
    AddAggregateLags,
    AddConstFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    CausalMeanValueImputation,
    DummyValueImputation,
    LastValueImputation,
    LeavesMissingValues,
    MeanValueImputation,
    MissingValueImputation,
    RollingMeanValueImputation,
    target_transformation_length,
)
from .field import (
    RemoveFields,
    RenameFields,
    SelectFields,
    SetField,
    SetFieldIfNotPresent,
)
from .sampler import (
    BucketInstanceSampler,
    ContinuousTimePointSampler,
    ContinuousTimeUniformSampler,
    ContinuousTimePredictionSampler,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    TestSplitSampler,
    ValidationSplitSampler,
    UniformSplitSampler,
)
from .split import (
    CanonicalInstanceSplitter,
    ContinuousTimeInstanceSplitter,
    InstanceSplitter,
    shift_timestamp,
)

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)

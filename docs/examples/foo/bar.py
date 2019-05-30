# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
"""Dummy utilities."""

__all__ = [
    'skipgram_batch']

import mxnet as mx
import numpy as np
import gluonts as ts

def skipgram_batch(centers, contexts, num_tokens, dtype, index_dtype):
    """Create a batch for SG training objective."""
    contexts = mx.nd.array(contexts[2], dtype=index_dtype)
    indptr = mx.nd.arange(len(centers) + 1)
    centers = mx.nd.array(centers, dtype=index_dtype)
    centers_csr = mx.nd.sparse.csr_matrix(
        (mx.nd.ones(centers.shape), centers, indptr), dtype=dtype,
        shape=(len(centers), num_tokens))
    return centers_csr, contexts, centers

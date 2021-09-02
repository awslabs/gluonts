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

import tempfile
from gluonts.nursery.ts_embeddings.run import main


def test_ts_embedding_smoke():
    with tempfile.TemporaryDirectory() as tmp_dir:
        main(
            f"""
        --fast_dev_run
        --encoder_path={tmp_dir}/encoder.pt
        --dataset_name=traffic
        --ts_len=14000
        --compared_length=500
        --lr=0.005
        --loss_temperature=0.2
        --batch_size=16
        --multivar_dim=1
        --channels=2
        --out_channels=4
        --depth=3
        """.split()
        )

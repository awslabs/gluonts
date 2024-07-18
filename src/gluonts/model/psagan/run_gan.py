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

import logging
from pathlib import Path

from gluonts.dataset.common import load_datasets

# from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.psagan._estimator import syntheticEstimator
from gluonts.model.psagan._trainer import Trainer

# from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)
# dataset = get_dataset("electricity")
dataset = load_datasets(
    metadata=Path(
        "/Users/pauljeha/Downloads/electricity_nips_scaled-stretch-len-5/metadata"
    ),
    train=Path(
        "/Users/pauljeha/Downloads/electricity_nips_scaled-stretch-len-5/train"
    ),
    test=Path(
        "/Users/pauljeha/Downloads/electricity_nips_scaled-stretch-len-5/test"
    ),
)

print(dataset)
estimator = syntheticEstimator(
    trainer=Trainer(
        num_epochs=20,
        lr_generator=0.00005,
        lr_discriminator=0.00005,
        schedule=[5, 10, 15],
        nb_step_discrim=2,
        EMA_value=0.2,
        nb_epoch_fade_in_new_layer=2,
        use_loss="lsgan",
        momment_loss=1,
        scaling="NoScale",
        scaling_penalty=0,
        encoder_network_factor=0,
        encoder_network_path="/Users/pauljeha/Documents/MyTsProject/gluon-ts-gan/src/gluonts/model/syntheticTransformer/CNN_embedder/m4_hourly",
        LARS=False,
    ),
    freq=dataset.metadata.freq,
    batch_size=512,
    num_batches_per_epoch=100,
    target_len=2 ** 6,
    nb_features=15 + 10 + 10,
    ks_conv=3,
    key_features=1,
    value_features=1,
    ks_value=1,
    ks_query=1,
    ks_key=1,
    path_to_pretrain=False,
    scaling="NoScale",
    cardinality=[370],
    embedding_dim=10,
    self_attention=True,
    pos_enc_dimension=20,
    context_length=64,
)
print(estimator.trainer.schedule)
net = estimator.train(dataset.train)
net.serialize(Path("/Users/pauljeha/Documents/MyTsProject/gluon-ts-gan"))
# pred = net.predict(dataset.train)
# for i in pred:
#     print(i)

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

from typing import cast, List, Literal, Optional
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from lightkit.data import DataLoader
from torch import nn
from torch.utils.data import TensorDataset
from tsbench.config import Config, EnsembleConfig
from tsbench.evaluations.tracking import EnsembleTracker
from tsbench.surrogate.torch.deepset import DeepSetModel
from ._base import OutputNormalization, Surrogate
from ._factory import register_ensemble_surrogate
from .torch import DeepSetLightningModule, ListMLELoss
from .transformers import EnsembleConfigTransformer


@register_ensemble_surrogate("deepset")
class DeepSetSurrogate(Surrogate[EnsembleConfig]):
    """
    The DeepSet surrogate is similar to the MLP surrogate but makes predictions
    for ensembles rather than single models.

    Currently, it does not support the use of dataset features.
    """

    trainer_: pl.Trainer
    models_: List[nn.Module]

    def __init__(
        self,
        tracker: EnsembleTracker,
        objective: Literal["regression", "ranking"] = "ranking",
        discount: Optional[
            Literal["logarithmic", "linear", "quadratic"]
        ] = "linear",
        hidden_layer_sizes: Optional[List[int]] = None,
        weight_decay: float = 0.01,
        dropout: float = 0.0,
        predict: Optional[List[str]] = None,
        output_normalization: OutputNormalization = None,
        impute_simulatable: bool = False,
    ):
        """
        Args:
            tracker: A tracker that can be used to impute latency and number of model parameters
                into model performances. Also, it is required for some input features.
            objective: The optimization objective for the XGBoost estimators.
            discount: The discount to apply for the ranking loss. If provided, it focuses on
                correctly predicting the top values.
            hidden_layer_sizes: The dimensions of the hidden layers. Defaults to two hidden layers
                of size 32.
            weight_decay: The weight decay to apply during optimization.
            dropout: The dropout probability of dropout layers applied after every activation
                function.
            predict: The metrics to predict. All if not provided.
            output_normalization: The type of normalization to apply to the features of each
                dataset independently. `None` applies no normalization, "quantile" applies quantile
                normalization, and "standard" transforms data to have zero mean and unit variance.
            impute_simulatable: Whether the tracker should impute latency and number of model
                parameters into the returned performance object.
        """

        super().__init__(
            tracker, predict, output_normalization, impute_simulatable
        )

        self.use_ranking = objective == "ranking"
        self.hidden_layer_sizes = hidden_layer_sizes or [32, 32]
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.config_transformer = EnsembleConfigTransformer()

        if objective == "regression":
            self.loss = nn.MSELoss()
        elif objective == "ranking":
            self.loss = ListMLELoss(discount=discount)

    @property
    def required_cpus(self) -> int:
        return 4

    def _fit(
        self, X: List[Config[EnsembleConfig]], y: npt.NDArray[np.float32]
    ) -> None:
        # Fit transformers to infer dimensionality
        X_numpy_list = self.config_transformer.fit_transform(X)
        X_numpy = np.concatenate(X_numpy_list)
        X_lengths_numpy = np.array([x.shape[0] for x in X_numpy_list])

        input_dim = len(self.config_transformer.feature_names_)
        output_dim = y.shape[1]

        # For initializing data, we prepare group IDs for the datasets
        mapping = {d: i for i, d in enumerate({x.dataset for x in X})}

        # For each output variable, we need to train a separate model
        self.models_ = []
        for i in range(output_dim):
            model = self._init_model(input_dim)
            module = DeepSetLightningModule(
                model, self.loss, self.weight_decay
            )

            # Train on output variable i
            dataset = TensorDataset(
                torch.from_numpy(X_numpy).float(),
                torch.from_numpy(X_lengths_numpy).long(),
                torch.from_numpy(y[:, i : i + 1]).float(),
                torch.as_tensor(
                    [mapping[x.dataset] for x in X], dtype=torch.long
                ),
            )
            train_loader = DataLoader(dataset, batch_size=len(dataset))
            self._trainer.fit(module, train_dataloaders=train_loader)

            # Add to models
            self.models_.append(model)

    def _predict(
        self, X: List[Config[EnsembleConfig]]
    ) -> npt.NDArray[np.float32]:
        # Get data
        X_numpy_list = self.config_transformer.transform(X)
        X_numpy = np.concatenate(X_numpy_list)
        X_lengths_numpy = np.array([x.shape[0] for x in X_numpy_list])

        dataset = TensorDataset(
            torch.from_numpy(X_numpy).float(),
            torch.from_numpy(X_lengths_numpy).long(),
        )
        test_loader = DataLoader(dataset, batch_size=len(dataset))

        # Run prediction
        predictions = []
        for model in self.models_:
            module = DeepSetLightningModule(model, self.loss)
            out = cast(
                List[torch.Tensor], self._trainer.predict(module, test_loader)
            )
            predictions.append(out[0].numpy())

        return np.concatenate(predictions, axis=-1)

    @property
    def _trainer(self) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=1000,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            gpus=int(torch.cuda.is_available()),
        )

    def _init_model(self, input_dim: int) -> nn.Module:
        return DeepSetModel(
            input_dim,
            self.hidden_layer_sizes[-1],
            1,
            self.hidden_layer_sizes,
            self.hidden_layer_sizes,
            self.dropout,
        )

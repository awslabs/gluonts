from gluonts.distribution.lowrank_gp import LowrankGPOutput
from typing import List

from gluonts.distribution import bijection
from gluonts.distribution.multivariate_independent_gaussian import (
    MultivariateIndependentGaussianOutput,
)
from gluonts.distribution.transformed_distribution import (
    TransformedDistribution,
)
from gluonts.distribution import (
    LowrankMultivariateGaussianOutput,
    MultivariateGaussianOutput,
)
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.multivariate.hyperparams import Hyperparams
from gluonts.trainer import Trainer


def freq_to_context_length(freq: str, prediction_length):
    if "H" in freq:
        context_length = 3 * prediction_length
    elif "D" in freq:
        context_length = 3 * prediction_length
    else:
        context_length = prediction_length
    return context_length


def trainer_from_params(
    params: Hyperparams,
    target_dim: int,
    low_rank: bool = True,
    hybridize: bool = None,
):
    # find a batch_size so that 1024 examples are used for SGD and cap the value in [8, 32]
    batch_size = params.batch_size
    if target_dim > 1000 or not low_rank:
        # avoid OOM
        batch_size = 4
    # batch_size = 512 // target_dim
    # batch_size = min(max(8, 1024 // max(batch_size, 1)), 32)
    # if not low_rank:
    #    # avoid OOM
    #    batch_size = 4

    return Trainer(
        epochs=params.epochs,
        batch_size=batch_size,  # todo make it dependent from dimension
        learning_rate=params.learning_rate
        if low_rank
        else params.learning_rate_fullrank,
        minimum_learning_rate=params.minimum_learning_rate,
        patience=params.patience,
        num_batches_per_epoch=params.num_batches_per_epoch,
        hybridize=hybridize if hybridize is not None else params.hybridize,
    )


class LowrankMultivariateGaussianOutputTransformed(
    LowrankMultivariateGaussianOutput
):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def distribution(self, dist_args, scale=None):
        base_dist = super(
            LowrankMultivariateGaussianOutputTransformed, self
        ).distribution(dist_args, scale)
        if isinstance(self.transform, List):
            return TransformedDistribution(base_dist, *self.transform)
        else:
            return TransformedDistribution(base_dist, self.transform)


class GPLowrankMultivariateGaussianOutputTransformed(LowrankGPOutput):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def distribution(self, dist_args, scale=None, dim=None):
        base_dist = super(
            GPLowrankMultivariateGaussianOutputTransformed, self
        ).distribution(dist_args, scale, dim)
        if isinstance(self.transform, List):
            return TransformedDistribution(base_dist, *self.transform)
        else:
            return TransformedDistribution(base_dist, self.transform)


class MultivariateGaussianOutputTransformed(MultivariateGaussianOutput):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def distribution(self, dist_args, scale=None):
        base_dist = super(
            MultivariateGaussianOutputTransformed, self
        ).distribution(dist_args, scale)
        if isinstance(self.transform, List):
            return TransformedDistribution(base_dist, *self.transform)
        else:
            return TransformedDistribution(base_dist, self.transform)


class MultivariateIndependentGaussianOutputTransformed(
    MultivariateIndependentGaussianOutput
):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def distribution(self, dist_args, scale=None):
        base_dist = super(
            MultivariateIndependentGaussianOutputTransformed, self
        ).distribution(dist_args, scale)
        if isinstance(self.transform, List):
            return TransformedDistribution(base_dist, *self.transform)
        else:
            return TransformedDistribution(base_dist, self.transform)


def distr_output_from_params(
    target_dim, diagonal_only, transform, low_rank, params
):
    if not low_rank:
        likelihood = MultivariateGaussianOutputTransformed(
            transform, dim=target_dim
        )
    else:
        if diagonal_only:
            likelihood = MultivariateIndependentGaussianOutputTransformed(
                transform, dim=target_dim
            )
        else:
            likelihood = LowrankMultivariateGaussianOutputTransformed(
                transform,
                dim=target_dim,
                rank=min(params.rank, target_dim) if not diagonal_only else 1,
            )
    return likelihood


def make_multivariate_estimator(
    low_rank: bool,
    diagonal_only: bool,
    cdf: bool = True,
    rnn: bool = True,
    scaling: bool = False,
):
    def make_model(
        freq: str, prediction_length: int, target_dim: int, params: Hyperparams
    ):
        transform = [bijection.identity]

        distr_output = distr_output_from_params(
            target_dim=target_dim,
            diagonal_only=diagonal_only,
            transform=transform,
            low_rank=low_rank,
            params=params,
        )

        context_length = freq_to_context_length(
            freq=freq, prediction_length=prediction_length
        )

        estimator = DeepVAREstimator(
            target_dim=target_dim,
            num_cells=params.num_cells,
            num_layers=params.num_layers,
            dropout_rate=params.dropout_rate,
            prediction_length=prediction_length,
            context_length=context_length,
            cell_type=params.cell_type if rnn else "time-distributed",
            freq=freq,
            pick_incomplete=False,
            distr_output=distr_output,
            conditioning_length=params.conditioning_length,
            trainer=trainer_from_params(
                params=params,
                target_dim=target_dim,
                low_rank=low_rank,
                hybridize=params.hybridize,
            ),
            scaling=scaling,
            use_copula=cdf,
            lags_seq=params.lags_seq,
        )
        return estimator

    return make_model


def make_gp_estimator(
    cdf: bool = True, rnn: bool = True, scaling: bool = False
):
    def _make_gp_estimator(
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Hyperparams = Hyperparams(),
    ):

        context_length = freq_to_context_length(
            freq=freq, prediction_length=prediction_length
        )

        transform = [bijection.identity]

        distr_output = GPLowrankMultivariateGaussianOutputTransformed(
            transform,
            dim=target_dim,
            rank=min(params.rank, target_dim),
            dropout_rate=params.dropout_rate,
        )

        return GPVAREstimator(
            target_dim=target_dim,
            num_cells=params.num_cells,
            num_layers=params.num_layers,
            dropout_rate=params.dropout_rate,
            prediction_length=prediction_length,
            context_length=context_length,
            cell_type=params.cell_type if rnn else "time-distributed",
            target_dim_sample=params.target_dim_sample,
            lags_seq=params.lags_seq,
            pick_incomplete=False,
            conditioning_length=params.conditioning_length,
            scaling=scaling,
            freq=freq,
            use_copula=cdf,
            distr_output=distr_output,
            trainer=trainer_from_params(
                params=params,
                target_dim=target_dim,
                hybridize=params.hybridize,
            ),
        )

    return _make_gp_estimator


models_dict = {
    "LSTMIndScaling": make_multivariate_estimator(
        low_rank=True, diagonal_only=True, cdf=False, scaling=True
    ),
    "LSTMInd": make_multivariate_estimator(
        low_rank=True, diagonal_only=True, cdf=False
    ),
    "LSTMFRScaling": make_multivariate_estimator(
        low_rank=False, diagonal_only=False, cdf=False, scaling=True
    ),
    "LSTMFR": make_multivariate_estimator(
        low_rank=False, diagonal_only=False, cdf=False
    ),
    "LSTMCOP": make_multivariate_estimator(
        low_rank=True, diagonal_only=False, cdf=True
    ),
    "GPCOP": make_gp_estimator(cdf=True, rnn=True),
    "GP": make_gp_estimator(cdf=False, rnn=True),
    "GPScaling": make_gp_estimator(cdf=False, rnn=True, scaling=True),
}

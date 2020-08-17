from typing import Tuple, List
from pandas.tseries.frequencies import to_offset
import numpy as np
import torch
from torch import nn
from torch_extensions.ops import batch_diag_matrix, make_block_diagonal
from utils.utils import one_hot
from gluonts.time_feature import (
    TimeFeature,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    WeekOfYear,
    MonthOfYear,
)


class ISSM(nn.Module):
    def __init__(self):
        super().__init__()
        # register a dummy parameter such that model.to(device) reaches it; used as property.
        self._dummy = torch.nn.Parameter(torch.empty(1), requires_grad=False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    @property
    def n_state(self):
        raise NotImplementedError("")

    @property
    def n_obs(self):
        raise NotImplementedError("")

    def A(self, seasonal_indicators) -> torch.Tensor:
        raise NotImplementedError("")

    def C(self, seasonal_indicators) -> torch.Tensor:
        raise NotImplementedError("")

    def R_diag_projector(self, seasonal_indicators) -> torch.Tensor:
        raise NotImplementedError(
            "Should return a matrix with dims: (n_state, n_params)."
            "May have leading time and batch dimensions."
            "n_params are the the learnt parameters."
            "level: 1, level+trend:2, seasonality: 1."
        )

    def forward(self, seasonal_indicators):
        return (
            self.A(seasonal_indicators),
            self.C(seasonal_indicators),
            self.R_diag_projector(seasonal_indicators),
        )


class LevelISSM(ISSM):
    def __init__(self):
        super().__init__()

    @property
    def n_state(self):
        return 1

    @property
    def n_obs(self):
        return 1

    def A(self, seasonal_indicators):
        A = torch.eye(self.n_state)
        for i in range(seasonal_indicators.ndim - 1):
            A = A[None, ...]
        A = A.repeat(seasonal_indicators.shape[:-1] + (1, 1,))
        return A.to(self.dtype).to(self.device)

    def C(self, seasonal_indicators):
        C = torch.ones(
            seasonal_indicators.shape[:-1] + (self.n_obs, self.n_state,)
        )
        return C.to(self.dtype).to(self.device)

    def R_diag_projector(self, seasonal_indicators):
        # R_diag_projector = torch.ones(seasonal_indicators.shape[:-1] + (self.n_state,))
        R_diag_projector = batch_diag_matrix(
            torch.ones(seasonal_indicators.shape[:-1] + (self.n_state,))
        )
        return R_diag_projector.to(self.dtype).to(self.device)


class LevelTrendISSM(LevelISSM):
    def __init__(self):
        super().__init__()

    @property
    def n_state(self):
        return 2

    @property
    def n_obs(self):
        return 1

    def A(self, seasonal_indicators):
        A = torch.diag(torch.ones(2), diagonal=0) + torch.diag(
            torch.ones(1), diagonal=1
        )
        for i in range(seasonal_indicators.ndim - 1):
            A = A[None, ...]
        A = A.repeat(seasonal_indicators.shape[:-1] + (1, 1,))
        return A.to(self.dtype).to(self.device)

    def C(self, seasonal_indicators):
        C = torch.ones(
            seasonal_indicators.shape[:-1] + (self.n_obs, self.n_state,)
        )
        return C.to(self.dtype).to(self.device)

    def R_diag_projector(self, seasonal_indicators):
        # R_diag_projector = torch.ones(seasonal_indicators.shape[:-1] + (self.n_state,))
        R_diag_projector = batch_diag_matrix(
            torch.ones(seasonal_indicators.shape[:-1] + (self.n_state,))
        )
        return R_diag_projector.to(self.dtype).to(self.device)


class SeasonalityISSM(ISSM):
    def __init__(self, n_seasons):
        super().__init__()
        self.n_seasons = n_seasons

    @property
    def n_state(self):
        return self.n_seasons

    @property
    def n_obs(self):
        return 1

    def A(self, seasonal_indicators):
        A = torch.eye(self.n_state)
        for i in range(seasonal_indicators.ndim - 1):  # particle and batch dim
            A = A[None, ...]
        A = A.repeat(seasonal_indicators.shape[:-1] + (1, 1))
        return A.to(self.dtype).to(self.device)

    def C(self, seasonal_indicators):
        C = one_hot(seasonal_indicators, num_classes=self.n_state,)[
            ..., None, :
        ]
        return C.to(self.dtype).to(self.device)

    def R_diag_projector(self, seasonal_indicators):
        R_diag_projector = one_hot(
            seasonal_indicators, num_classes=self.n_state,
        )[..., None]
        return R_diag_projector.to(self.dtype).to(self.device)


class CompositeISSM(ISSM):
    def __init__(self, seasonal_issms: List[SeasonalityISSM], add_trend):
        super(CompositeISSM, self).__init__()
        self.nonseasonal_issm = (
            LevelISSM() if add_trend is False else LevelTrendISSM()
        )
        self.seasonal_issms = seasonal_issms
        for idx, issm in enumerate(self.seasonal_issms):
            super().add_module(f"seasonal_issm_{idx}", issm)

    @classmethod
    def get_from_freq(cls, freq: str, add_trend):
        offset = to_offset(freq)

        seasonal_issms: List[SeasonalityISSM] = []

        if offset.name == "M":
            seasonal_issms = [
                SeasonalityISSM(n_seasons=12)  # month-of-year seasonality
            ]
        elif offset.name == "W-SUN":
            seasonal_issms = [
                SeasonalityISSM(n_seasons=53)  # week-of-year seasonality
            ]
        elif offset.name == "D":
            seasonal_issms = [
                SeasonalityISSM(n_seasons=7)  # day-of-week seasonality
            ]
        elif offset.name == "B":
            seasonal_issms = [
                SeasonalityISSM(n_seasons=7)  # day-of-week seasonality
            ]
        elif offset.name == "H":
            seasonal_issms = [
                SeasonalityISSM(n_seasons=24),  # hour-of-day seasonality
                SeasonalityISSM(n_seasons=7),  # day-of-week seasonality
            ]
        elif offset.name == "T":
            seasonal_issms = [
                SeasonalityISSM(n_seasons=60),  # minute-of-hour seasonality
                SeasonalityISSM(n_seasons=24),  # hour-of-day seasonality
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return cls(seasonal_issms=seasonal_issms, add_trend=add_trend)

    @classmethod
    def seasonal_features(cls, freq: str) -> List[TimeFeature]:
        offset = to_offset(freq)
        if offset.name == "M":
            return [MonthOfYear(normalized=False)]
        elif offset.name == "W-SUN":
            return [WeekOfYear(normalized=False)]
        elif offset.name == "D":
            return [DayOfWeek(normalized=False)]
        elif offset.name == "B":
            return [DayOfWeek(normalized=False)]
        elif offset.name == "H":
            return [HourOfDay(normalized=False), DayOfWeek(normalized=False)]
        elif offset.name == "T":
            return [
                MinuteOfHour(normalized=False),
                HourOfDay(normalized=False),
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return []

    def forward(
        self, seasonal_indicators: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seasonal_issm_params = [
            issm(seasonal_indicators[..., idx : idx + 1])
            for idx, issm in enumerate(self.seasonal_issms)
        ]
        nonseasonal_issm_params = self.nonseasonal_issm(seasonal_indicators)
        A, C, R_diag_projector = zip(
            nonseasonal_issm_params, *seasonal_issm_params
        )

        return (
            make_block_diagonal(A),
            torch.cat(C, dim=-1),
            make_block_diagonal(R_diag_projector),
        )


if __name__ == "__main__":
    import mxnet as mx
    from gluonts.model.deepstate.issm import CompositeISSM as GtsCompositeISSM
    from data.gluonts_nips_datasets.gluonts_nips_datasets import (
        create_loaders,
        transform_gluonts_to_pytorch,
        get_cardinalities,
        get_dataset,
    )
    from experiments.base_config import TimeFeatType

    dataset_names = [
        "exchange_rate_nips",
        "electricity_nips",
        "traffic_nips",
        "solar_nips",
        "wiki-rolling_nips",
    ]
    for dataset_name in dataset_names:
        for add_trend in [True, False]:
            dataset = get_dataset(dataset_name)
            (
                train_loader,
                val_loader,
                inference_loader,
                input_transforms,
            ) = create_loaders(
                dataset=dataset,
                batch_size=100,
                past_length=28,
                n_steps_forecast=14,
            )
            freq = dataset.metadata.freq
            cardinalities = get_cardinalities(
                dataset=dataset, add_trend=add_trend
            )
            batch = next(iter(inference_loader))

            gts_seasonal_indicators = mx.ndarray.concat(
                batch["past_seasonal_indicators"],
                batch["future_seasonal_indicators"],
                dim=1,
            )
            gts_issm = GtsCompositeISSM.get_from_freq(
                freq=freq, add_trend=add_trend
            )
            (
                emission_coeff,
                transition_coeff,
                innovation_coeff,
            ) = gts_issm.get_issm_coeff(gts_seasonal_indicators)

            data = transform_gluonts_to_pytorch(
                batch=batch,
                device="cuda",
                dtype=torch.float32,
                time_features=TimeFeatType.seasonal_indicator,
                **cardinalities,
            )

            issm = CompositeISSM.get_from_freq(
                freq=freq, add_trend=add_trend
            ).to("cuda")
            A, C, R_diag_projector = issm(data["seasonal_indicators"])

            assert np.all(
                A.transpose(0, 1).cpu().numpy() == transition_coeff.asnumpy()
            )
            assert np.all(
                C.transpose(0, 1).cpu().numpy() == emission_coeff.asnumpy()
            )
            # we have a projection matrix and do a matvec product with the params
            # in gluonts, there is just a single parameter, thus they have just a vector.
            assert np.all(
                R_diag_projector.transpose(0, 1).sum(dim=-1).cpu().numpy()
                == innovation_coeff.asnumpy()
            )

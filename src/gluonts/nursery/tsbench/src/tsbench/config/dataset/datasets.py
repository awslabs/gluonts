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

# pylint: disable=too-many-lines
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Dict, List, Tuple
import numpy as np
import pandas as pd
from ._factory import register_dataset
from .preprocessing import ConstantTargetFilter, Filter
from .sources import (
    GluonTsDatasetConfig,
    KaggleDatasetConfig,
    M3DatasetConfig,
    MonashDatasetConfig,
)


@register_dataset
@dataclass(frozen=True)
class ExchangeRateDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `exchange_rate_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "exchange_rate"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _gluonts_name(self) -> str:
        return "exchange_rate_nips"


@register_dataset
@dataclass(frozen=True)
class ElectricityDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `electricity_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "electricity"

    @property
    def max_training_time(self) -> int:
        return 14400


@register_dataset
@dataclass(frozen=True)
class SolarDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `solar_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "solar"

    @property
    def max_training_time(self) -> int:
        return 7200

    @property
    def _gluonts_name(self) -> str:
        return "solar_nips"


@register_dataset
@dataclass(frozen=True)
class WikiDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `wiki-rolling_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "wiki"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def _gluonts_name(self) -> str:
        return "wiki-rolling_nips"


@register_dataset
@dataclass(frozen=True)
class TaxiDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `taxi_30min` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "taxi"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def _gluonts_name(self) -> str:
        return "taxi_30min"


@register_dataset
@dataclass(frozen=True)
class M3MonthlyDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_monthly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_monthly"

    @property
    def max_training_time(self) -> int:
        return 7200


@register_dataset
@dataclass(frozen=True)
class M3QuarterlyDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_quarterly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_quarterly"

    @property
    def max_training_time(self) -> int:
        return 3600


@register_dataset
@dataclass(frozen=True)
class M3YearlyDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_yearly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_yearly"

    @property
    def max_training_time(self) -> int:
        return 3600


@register_dataset
@dataclass(frozen=True)
class M3OtherDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_other` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_other"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def has_time_features(self) -> bool:
        return False


@register_dataset
@dataclass(frozen=True)
class M4HourlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_hourly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_hourly"

    @property
    def max_training_time(self) -> int:
        return 7200

    @property
    def has_time_features(self) -> bool:
        return False


@register_dataset
@dataclass(frozen=True)
class M4DailyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_daily` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_daily"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def has_time_features(self) -> bool:
        return False


@register_dataset
@dataclass(frozen=True)
class M4WeeklyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_weekly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_weekly"

    @property
    def max_training_time(self) -> int:
        return 7200

    @property
    def has_time_features(self) -> bool:
        return False


@register_dataset
@dataclass(frozen=True)
class M4MonthlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_monthly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_monthly"

    @property
    def max_training_time(self) -> int:
        return 28800

    @property
    def has_time_features(self) -> bool:
        return False


@register_dataset
@dataclass(frozen=True)
class M4QuarterlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_quarterly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_quarterly"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def has_time_features(self) -> bool:
        return False


@register_dataset
@dataclass(frozen=True)
class M4YearlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_yearly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_yearly"

    @property
    def max_training_time(self) -> int:
        return 7200

    @property
    def has_time_features(self) -> bool:
        return False


@register_dataset
@dataclass(frozen=True)
class M5DatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m5` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m5"

    @property
    def max_training_time(self) -> int:
        return 28800

    def _materialize(self, directory: Path, regenerate: bool = False) -> None:
        shutil.copytree(
            Path.home() / ".mxnet" / "gluon-ts" / "datasets" / "m5",
            directory / "m5",
        )
        super()._materialize(directory, regenerate=True)


@register_dataset
@dataclass(frozen=True)
class TourismMonthlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `tourism_monthly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "tourism_monthly"

    @property
    def max_training_time(self) -> int:
        return 7200


@register_dataset
@dataclass(frozen=True)
class TourismQuarterlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `tourism_quarterly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "tourism_quarterly"

    @property
    def max_training_time(self) -> int:
        return 3600


@register_dataset
@dataclass(frozen=True)
class TourismYearlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `tourism_yearly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "tourism_yearly"

    @property
    def max_training_time(self) -> int:
        return 3600


@register_dataset
@dataclass(frozen=True)
class NN5DatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `nn5_daily_without_missing` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "nn5"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _gluonts_name(self) -> str:
        return "nn5_daily_without_missing"


@register_dataset
@dataclass(frozen=True)
class LondonSmartMetersDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "London Smart Meters".
    """

    @classmethod
    def name(cls) -> str:
        return "london_smart_meters"

    @property
    def max_training_time(self) -> int:
        return 28800

    @property
    def _file(self) -> str:
        return "london_smart_meters_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "4656091"

    @property
    def _prediction_length(self) -> int:
        return 48


@register_dataset
@dataclass(frozen=True)
class WindFarmsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Wind Farms".
    """

    @classmethod
    def name(cls) -> str:
        return "wind_farms"

    @property
    def max_training_time(self) -> int:
        return 28800

    def _filters(self, prediction_length: int) -> List[Filter]:
        return [
            ConstantTargetFilter(prediction_length, required_length=100000)
        ]

    @property
    def _file(self) -> str:
        return "wind_farms_minutely_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "4654858"

    @property
    def _prediction_length(self) -> int:
        return 60


@register_dataset
@dataclass(frozen=True)
class CarPartsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Car Parts".
    """

    @classmethod
    def name(cls) -> str:
        return "car_parts"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "car_parts_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "4656021"

    @property
    def _prediction_length(self) -> int:
        return 12


@register_dataset
@dataclass(frozen=True)
class DominickDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Dominick".
    """

    @classmethod
    def name(cls) -> str:
        return "dominick"

    @property
    def max_training_time(self) -> int:
        return 28800

    @property
    def has_time_features(self) -> bool:
        return False

    @property
    def _file(self) -> str:
        return "dominick_dataset.zip"

    @property
    def _record(self) -> str:
        return "4654802"

    @property
    def _prediction_length(self) -> int:
        return 8

    @property
    def _prediction_length_multiplier(self) -> int:
        return 1


@register_dataset
@dataclass(frozen=True)
class FredMdDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Federal Reserve Economic Dataset".
    """

    @classmethod
    def name(cls) -> str:
        return "fred_md"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "fred_md_dataset.zip"

    @property
    def _record(self) -> str:
        return "4654833"

    @property
    def _prediction_length(self) -> int:
        return 12


@register_dataset
@dataclass(frozen=True)
class SanFranciscoTrafficDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "San Francisco Traffic".
    """

    @classmethod
    def name(cls) -> str:
        return "san_francisco_traffic"

    @property
    def max_training_time(self) -> int:
        return 28800

    @property
    def _file(self) -> str:
        return "traffic_hourly_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656132"

    @property
    def _prediction_length(self) -> int:
        return 48


@register_dataset
@dataclass(frozen=True)
class PedestrianCountDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Pedestrian Counts".
    """

    @classmethod
    def name(cls) -> str:
        return "pedestrian_count"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def _file(self) -> str:
        return "pedestrian_counts_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656626"

    @property
    def _prediction_length(self) -> int:
        return 48


@register_dataset
@dataclass(frozen=True)
class HospitalDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Hospitals".
    """

    @classmethod
    def name(cls) -> str:
        return "hospital"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "hospital_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656014"

    @property
    def _prediction_length(self) -> int:
        return 12


@register_dataset
@dataclass(frozen=True)
class CovidDeathsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "COVID Deaths".
    """

    @classmethod
    def name(cls) -> str:
        return "covid_deaths"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "covid_deaths_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656009"

    @property
    def _prediction_length(self) -> int:
        return 30


@register_dataset
@dataclass(frozen=True)
class KddCupDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "KDD Cup 2018".
    """

    @classmethod
    def name(cls) -> str:
        return "kdd_2018"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def _file(self) -> str:
        return "kdd_cup_2018_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "4656756"

    @property
    def _prediction_length(self) -> int:
        return 48


@register_dataset
@dataclass(frozen=True)
class CifDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "CIF 2016".
    """

    @classmethod
    def name(cls) -> str:
        return "cif_2016"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def has_time_features(self) -> bool:
        return False

    @property
    def _file(self) -> str:
        return "cif_2016_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656042"

    @property
    def _prediction_length(self) -> int:
        return 12


@register_dataset
@dataclass(frozen=True)
class AustralianElectricityDemandDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Australian Electricity Demand".
    """

    @classmethod
    def name(cls) -> str:
        return "australian_electricity_demand"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def _file(self) -> str:
        return "australian_electricity_demand_dataset.zip"

    @property
    def _record(self) -> str:
        return "4659727"

    @property
    def _prediction_length(self) -> int:
        return 48


@register_dataset
@dataclass(frozen=True)
class BitcoinDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Bitcoin".
    """

    @classmethod
    def name(cls) -> str:
        return "bitcoin"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "bitcoin_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "5122101"

    @property
    def _prediction_length(self) -> int:
        return 30


@register_dataset
@dataclass(frozen=True)
class RideshareDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Rideshare".
    """

    @classmethod
    def name(cls) -> str:
        return "rideshare"

    @property
    def max_training_time(self) -> int:
        return 14400

    @property
    def _file(self) -> str:
        return "rideshare_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "5122232"

    @property
    def _prediction_length(self) -> int:
        return 48


@register_dataset
@dataclass(frozen=True)
class VehicleTripsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Vehicle Trips".
    """

    @classmethod
    def name(cls) -> str:
        return "vehicle_trips"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "vehicle_trips_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "5122537"

    @property
    def _prediction_length(self) -> int:
        return 30

    @property
    def _prediction_length_multiplier(self) -> int:
        return 1


@register_dataset
@dataclass(frozen=True)
class WeatherDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Weather".
    """

    @classmethod
    def name(cls) -> str:
        return "weather"

    @property
    def max_training_time(self) -> int:
        return 28800

    @property
    def _file(self) -> str:
        return "weather_dataset.zip"

    @property
    def _record(self) -> str:
        return "4654822"

    @property
    def _prediction_length(self) -> int:
        return 30


@register_dataset
@dataclass(frozen=True)
class TemperatureRainDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Temperature Rain".
    """

    @classmethod
    def name(cls) -> str:
        return "temperature_rain"

    @property
    def max_training_time(self) -> int:
        return 28800

    @property
    def _file(self) -> str:
        return "temperature_rain_dataset_without_missing_values.zip"

    @property
    def _record(self) -> str:
        return "5129091"

    @property
    def _prediction_length(self) -> int:
        return 30


@register_dataset
@dataclass(frozen=True)
class M1YearlyDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "M1 Yearly".
    """

    @classmethod
    def name(cls) -> str:
        return "m1_yearly"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "m1_yearly_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656193"

    @property
    def _prediction_length(self) -> int:
        return 6


@register_dataset
@dataclass(frozen=True)
class M1QuarterlyDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "M1 Quarterly".
    """

    @classmethod
    def name(cls) -> str:
        return "m1_quarterly"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "m1_quarterly_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656154"

    @property
    def _prediction_length(self) -> int:
        return 8


@register_dataset
@dataclass(frozen=True)
class M1MonthlyDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "M1 Monthly".
    """

    @classmethod
    def name(cls) -> str:
        return "m1_monthly"

    @property
    def max_training_time(self) -> int:
        return 3600

    @property
    def _file(self) -> str:
        return "m1_monthly_dataset.zip"

    @property
    def _record(self) -> str:
        return "4656159"

    @property
    def _prediction_length(self) -> int:
        return 18


@register_dataset
@dataclass(frozen=True)
class RossmannDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Rossmann Store Sales" Kaggle
    competition.
    """

    @classmethod
    def name(cls) -> str:
        return "rossmann"

    @property
    def max_training_time(self) -> int:
        return 7200

    @property
    def _link(self) -> str:
        return "https://www.kaggle.com/c/rossmann-store-sales"

    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "train.csv"))
        stores = cast(pd.DataFrame, pd.read_csv(path / "store.csv"))

        # Generate GluonTS dataset
        metadata = {
            "freq": "D",
            "prediction_length": 48,
            "feat_static_cat": [
                {
                    "name": "store",
                    "cardinality": len(stores),
                },
            ],
        }

        series = []
        for i, store_data in data.groupby("Store"):
            sorted_data = store_data.sort_values("Date")
            series.append(
                {
                    "item_id": int(i) - 1,
                    "start": sorted_data.Date.min(),
                    "target": sorted_data.Sales.to_list(),
                    "feat_static_cat": [
                        int(i) - 1,
                    ],
                }
            )

        return metadata, series


@register_dataset
@dataclass(frozen=True)
class CorporacionFavoritaDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Corporación Favorita" Kaggle
    competition.
    """

    @classmethod
    def name(cls) -> str:
        return "corporacion_favorita"

    @property
    def max_training_time(self) -> int:
        return 28800

    @property
    def _link(self) -> str:
        return "https://www.kaggle.com/c/favorita-grocery-sales-forecasting"

    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "train.csv"))
        stores = cast(pd.DataFrame, pd.read_csv(path / "stores.csv"))
        item_ids = np.sort(data.item_nbr.unique())

        # Generate GluonTS dataset
        metadata = {
            "freq": "D",
            "prediction_length": 16,
            "feat_static_cat": [
                {
                    "name": "store",
                    "cardinality": len(stores),
                },
                {
                    "name": "item",
                    "cardinality": len(item_ids),
                },
            ],
        }

        series = []
        for i, ((item, store_id), group_data) in enumerate(
            data.groupby(["item_nbr", "store_nbr"])
        ):
            item_id = np.where(item_ids == item)[0][0]
            sorted_data = group_data.sort_values("date")
            sales = pd.Series(
                sorted_data.unit_sales.to_numpy(),
                index=pd.DatetimeIndex(sorted_data.date),
            )
            series.append(
                {
                    "item_id": i,
                    "start": sorted_data.date.min(),
                    "target": sales.resample("D")
                    .first()
                    .fillna(value=0)
                    .to_list(),
                    "feat_static_cat": [
                        int(store_id) - 1,
                        int(item_id),
                    ],
                }
            )

        return metadata, series


@register_dataset
@dataclass(frozen=True)
class WalmartDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Walmart Recruiting" Kaggle competition.
    """

    @classmethod
    def name(cls) -> str:
        return "walmart"

    @property
    def max_training_time(self) -> int:
        return 7200

    @property
    def _link(self) -> str:
        return "https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting"

    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "train.csv"))
        department_ids = np.sort(data.Dept.unique())

        # Generate GluonTS dataset
        metadata = {
            "freq": "W",
            "prediction_length": 39,
            "feat_static_cat": [
                {
                    "name": "store",
                    "cardinality": len(
                        data.Store.unique()
                    ),  # pylint: disable=no-member
                },
                {
                    "name": "department",
                    "cardinality": len(department_ids),
                },
            ],
        }

        series = []
        # pylint: disable=no-member
        for i, ((store_id, department), group_data) in enumerate(
            data.groupby(["Store", "Dept"])
        ):
            department_id = np.where(department_ids == department)[0][0]
            sorted_data = group_data.sort_values("Date")
            series.append(
                {
                    "item_id": i,
                    "start": sorted_data.Date.min(),
                    "target": sorted_data.Weekly_Sales.to_list(),
                    "feat_static_cat": [
                        int(store_id) - 1,
                        int(department_id),
                    ],
                }
            )

        return metadata, series


@register_dataset
@dataclass(frozen=True)
class RestaurantDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Restaurant" Kaggle competition.
    """

    @classmethod
    def name(cls) -> str:
        return "restaurant"

    @property
    def max_training_time(self) -> int:
        return 7200

    @property
    def _link(self) -> str:
        return (
            "https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting"
        )

    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "air_visit_data.csv"))
        store_ids = np.sort(data.air_store_id.unique())

        # Generate GluonTS dataset
        metadata = {
            "freq": "D",
            "prediction_length": 39,
            "feat_static_cat": [
                {
                    "name": "restaurant",
                    "cardinality": len(store_ids),
                },
            ],
        }

        series = []
        # pylint: disable=no-member
        for i, (store, group_data) in enumerate(data.groupby("air_store_id")):
            store_id = np.where(store_ids == store)[0][0]
            sorted_data = group_data.sort_values("visit_date")
            visitors = pd.Series(
                sorted_data.visitors.to_numpy(),
                index=pd.DatetimeIndex(sorted_data.visit_date),
            )
            series.append(
                {
                    "item_id": i,
                    "start": sorted_data.visit_date.min(),
                    "target": visitors.resample("D")
                    .first()
                    .fillna(value=0)
                    .to_list(),
                    "feat_static_cat": [
                        int(store_id),
                    ],
                }
            )

        return metadata, series

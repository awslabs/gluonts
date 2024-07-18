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

import numpy as np
import pandas as pd

from gluonts.itertools import inverse

GH_URL = "https://github.com/robert-koch-institut/"


# Numbers of each Bundesland, according to the "Amtlicher Gemeindeschlüssel"
# https://de.wikipedia.org/wiki/Amtlicher_Gemeindeschl%C3%BCssel
BUNDESLAND_TO_ID = {
    "SH": 1,
    "HH": 2,
    "NI": 3,
    "HB": 4,
    "NW": 5,
    "HE": 6,
    "RP": 7,
    "BW": 8,
    "BY": 9,
    "SL": 10,
    "BE": 11,
    "BB": 12,
    "MV": 13,
    "SN": 14,
    "ST": 15,
    "TH": 16,
}
ID_TO_BUNDESLAND = inverse(BUNDESLAND_TO_ID)


def get_infections_data():
    """ """
    url = "/".join(
        [
            GH_URL,
            "SARS-CoV-2-Infektionen_in_Deutschland",
            "raw/main/",
            "Aktuell_Deutschland_SarsCov2_Infektionen.csv",
        ]
    )

    df = pd.read_csv(
        url,
        usecols=[
            "IdLandkreis",
            "Altersgruppe",
            "Meldedatum",
            "AnzahlFall",
            "AnzahlTodesfall",
        ],
    )
    # rename to Datum to be consistent with other data sources
    df.rename(columns={"Meldedatum": "Datum"}, inplace=True)
    df["Datum"] = df["Datum"].map(np.datetime64)

    # We are not interested in the communal level (LandKreis) but want data on
    # the state (Bundesland) level.
    bundesland_ids = df.pop("IdLandkreis") // 1000
    df["Bundesland"] = bundesland_ids.map(ID_TO_BUNDESLAND)

    # use age-groups as hospitalisation data
    df["Altersgruppe"] = df["Altersgruppe"].map(
        {
            "A00-A04": "00-04",
            "A05-A14": "05-14",
            "A15-A34": "15-34",
            "A35-A59": "35-59",
            "A60-A79": "60-79",
            "A80+": "80+",
        }
    )

    return (
        df.groupby(
            [
                "Bundesland",
                "Altersgruppe",
                "Datum",
            ]
        )
        .sum()
        .reset_index()
    )


def get_hospitalisation_data():
    url = "/".join(
        [
            GH_URL,
            "COVID-19-Hospitalisierungen_in_Deutschland",
            "raw/master/",
            "Aktuell_Deutschland_COVID-19-Hospitalisierungen.csv",
        ]
    )
    df = pd.read_csv(
        url,
        usecols=[
            "Datum",
            "Bundesland_Id",
            "Altersgruppe",
            "7T_Hospitalisierung_Faelle",
        ],
    )
    df.rename(
        columns={"7T_Hospitalisierung_Faelle": "Hospitalisierung"},
        inplace=True,
    )

    df["Datum"] = df["Datum"].map(np.datetime64)

    # there is aggregated level data, we want to ignore
    df = df[df["Bundesland_Id"] != 0]
    # Also remove aggregated data across age groups
    df = df[df["Altersgruppe"] != "00+"]

    df["Bundesland"] = df.pop("Bundesland_Id").map(ID_TO_BUNDESLAND)

    return df


def get_dataset():
    infections = get_infections_data()
    hospitalisations = get_hospitalisation_data()

    df = infections.merge(
        hospitalisations,
        on=["Bundesland", "Altersgruppe", "Datum"],
        how="outer",
    )

    for (bundesland, age_group), group in df.groupby(
        ["Bundesland", "Altersgruppe"]
    ):
        group.sort_values(["Datum"], inplace=True)

        yield {
            "date": group["Datum"].to_numpy(),
            "cases": group["AnzahlFall"].to_numpy(),
            "deaths": group["AnzahlTodesfall"].to_numpy(),
            "hospitalisations": group["Hospitalisierung"].to_numpy(),
            "state": bundesland,
            "age_group": age_group,
        }

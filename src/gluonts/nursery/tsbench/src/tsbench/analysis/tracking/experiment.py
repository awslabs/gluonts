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

import pickle
from io import BytesIO
from typing import Any, Dict, List
import pandas as pd
from gridfs import GridFS
from pymongo.database import Database


class SacredExperiment:
    """
    A sacred experiment describes a Sacred experiment stored in MongoDB and is
    retrieved from a Sacred Mongo client.
    """

    def __init__(self, info: Dict[str, Any], db: Database, gridfs: GridFS):
        """
        **Not to be used manually.**
        """
        self.info = info
        self.db = db
        self.fs = gridfs

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the experiment.
        """
        return self.info["config"]

    @property
    def artifacts(self) -> List[str]:
        """
        Returns the names of all artifacts associated with the experiment.
        """
        return [a["name"] for a in self.info["artifacts"]]

    def read_parquet(self, artifact: str) -> pd.DataFrame:
        """
        Reads the parquet file from the artifact with the specified name.

        Args:
            artifact: The name of the artifact.

        Returns:
            The parquet file loaded as data frame.
        """
        matches = [
            a["file_id"]
            for a in self.info["artifacts"]
            if a["name"] == artifact
        ]
        data = self.fs.get(matches[0]).read()
        return pd.read_parquet(BytesIO(data))

    def read_pickle(self, artifact: str) -> Any:
        """
        Reads the pickled file from the artifact with the specified name.

        Args:
            artifact: The name of the artifact.

        Returns:
            The data that was pickled.
        """
        matches = [
            a["file_id"]
            for a in self.info["artifacts"]
            if a["name"] == artifact
        ]
        data = self.fs.get(matches[0]).read()
        return pickle.loads(data)

    def delete(self) -> None:
        """
        Deletes the experiment by setting the associated experiment name to
        "Trash".
        """
        self.db.runs.update_one(
            {"_id": self.info["_id"]},
            {"$set": {"config.name": "Trash"}},
            upsert=False,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.info['config']})"

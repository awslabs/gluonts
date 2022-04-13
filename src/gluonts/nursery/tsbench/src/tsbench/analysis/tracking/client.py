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

from typing import Any, Dict, Iterator, List, Optional, Union
import gridfs
import pymongo
from .experiment import SacredExperiment


class SacredMongoClient:
    """
    The Sacred Mongo client allows to retrieve results from experiments tracked
    with Sacred from MongoDB.
    """

    base_query: Dict[str, Any]

    def __init__(
        self,
        experiment: Union[str, List[str]],
        host: str = "localhost",
        port: int = 27017,
        database: str = "sacred",
        only_completed: bool = True,
    ):
        """
        Args:
            experiment: The name(s) of the experiment(s) for which to query individual runs.
            only_completed: Whether only completed jobs should be returned.
            host: The hostname of the server where the MongoDB is running.
            port: The port where the MongoDB is reachable.
            database: The name of the database where Sacred experiments are stored.
        """
        # Initialize Mongo client
        client = pymongo.MongoClient(host, port)
        self.db = client[database]
        self.fs = gridfs.GridFS(self.db)

        # Initialize base query
        if isinstance(experiment, list):
            self.base_query = {"config.experiment": {"$in": experiment}}
        else:
            self.base_query = {"config.experiment": experiment}

        if only_completed:
            self.base_query.update({"status": "COMPLETED"})

    def query_one(
        self, config: Dict[str, Any], others: Optional[Dict[str, Any]] = None
    ) -> SacredExperiment:
        """
        Queries the experiments associated with the experiment set and returns
        the one matching the query.

        Args:
            config: The query items that apply to the configuration.
            others: Optional additional query options.

        Returns:
            The sacred experiment describing the experiment.
        """
        full_query = {
            **self.base_query,
            **{f"config.{k}": v for k, v in config.items()},
            **(others or {}),
        }
        assert (
            self.db.runs.count_documents(full_query) == 1
        ), "Query does not return a single experiment."

        info = self.db.runs.find_one(full_query)
        assert info is not None

        return SacredExperiment(info, self.db, self.fs)

    def query(
        self, config: Dict[str, Any], others: Optional[Dict[str, Any]] = None
    ) -> List[SacredExperiment]:
        """
        Queries the experiments associated with the experiment set and returns
        all matching the query.

        Args:
            config: The query items that apply to the configuration.
            others: Optional additional query options.

        Returns:
            The sacred experiments found via the query.
        """
        full_query = {
            **self.base_query,
            **{f"config.{k}": v for k, v in config.items()},
            **(others or {}),
        }
        infos = self.db.runs.find(full_query)
        return [SacredExperiment(info, self.db, self.fs) for info in infos]

    def __iter__(self) -> Iterator[SacredExperiment]:
        for info in self.db.runs.find(self.base_query):
            yield SacredExperiment(info, self.db, self.fs)

    def __len__(self) -> int:
        return self.db.runs.count_documents(self.base_query)

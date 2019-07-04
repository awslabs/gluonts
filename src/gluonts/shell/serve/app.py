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

# Standard library imports
import traceback
from typing import Tuple

# Third-party imports
import pydantic
from flask import Flask, Response, jsonify, request

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import Config as ForecastConfig


class InferenceRequest(pydantic.BaseModel):
    instances: list
    configuration: ForecastConfig


def make_app(predictor_factory, execution_params):
    flask = Flask('GluonTS scoring service')

    @flask.route('/ping')
    def ping() -> str:
        return ''

    @flask.errorhandler(Exception)
    def handle_error(error) -> Tuple[str, int]:
        return traceback.format_exc(), 500

    @flask.route("/execution-parameters")
    def execution_parameters() -> Response:
        return jsonify(execution_params)

    @flask.route('/invocations', methods=['POST'])
    def invocations() -> Response:
        predictor = predictor_factory(request.json)
        req = InferenceRequest.parse_obj(request.json)

        dataset = ListDataset(req.instances, predictor.freq)

        # create the forecasts
        forecasts = predictor.predict(
            dataset, num_eval_samples=req.configuration.num_eval_samples
        )

        return jsonify(
            predictions=[
                forecast.as_json_dict(req.configuration)
                for forecast in forecasts
            ]
        )

    return flask

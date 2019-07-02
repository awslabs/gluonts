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

import traceback

import pydantic
from flask import Flask, jsonify, request

from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import Config as ForecastConfig


class RequestPayload(pydantic.BaseModel):
    instances: list
    configuration: ForecastConfig


def make_app(predictor_factory, execution_params):
    app = Flask('GluonTS scoring service')

    @app.route('/ping')
    def ping():
        return ''

    @app.errorhandler(Exception)
    def handle_error(error):
        return traceback.format_exc(), 500

    @app.route("/execution-parameters")
    def execution_parameters():
        return jsonify(execution_params)

    @app.route('/invocations', methods=['POST'])
    def invocations():
        predictor = predictor_factory(request.json)
        req = RequestPayload.parse_obj(request.json)

        dataset = ListDataset(req.instances, predictor.freq)

        # create the forecasts
        forecasts = predictor.predict(
            dataset, num_eval_samples=req.configuration.num_eval_samples
        )

        return jsonify(
            predictions=list(map(req.configuration.process, forecasts))
        )

    return app

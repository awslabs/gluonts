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

from flask import Flask, jsonify, request

from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import (
    Config as ForecastConfig,
    SampleForecast,
)

def get_config():
    configuration = request.json['configuration']

    configuration.setdefault(
        "num_eval_samples", configuration.get("num_samples")
    )

    return ForecastConfig.parse_obj(configuration)


def make_app(predictor, execution_params):
    def make_response(config, forecast):
        if 'samples' in config.output_types:
            samples = []
            if isinstance(forecast, SampleForecast):
                samples = forecast.samples.tolist()
            yield 'samples', samples

        if 'mean' in config.output_types:
            yield 'mean', forecast.mean.tolist()

        if 'quantiles' in config.output_types:
            yield 'quantiles', {
                q: forecast.quantile(q).tolist() for q in config.quantiles
            }

    app = Flask('GluonTS scoring service')

    @app.route('/ping')
    def ping():
        return ''

    @app.route("/execution-parameters")
    def execution_parameters():
        return jsonify(execution_params)

    @app.route('/invocations', methods=['POST'])
    def invocations():
        try:
            config = get_config()
            dataset = ListDataset(request.json['instances'], predictor.freq)

            # create the forecasts
            forecasts = predictor.predict(
                dataset, num_eval_samples=config.num_eval_samples
            )

            # generate json output
            predictions = [
                dict(make_response(config, forecast)) for forecast in forecasts
            ]
            return jsonify(predictions=predictions)

        except Exception as error:
            return jsonify(error=traceback.format_exc()), 500

    return app

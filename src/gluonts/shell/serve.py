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
import multiprocessing
import os
import signal
import traceback
from typing import Optional, Any

# Third-party imports
import flask
from gunicorn.app.base import BaseApplication

# First-party imports
from gluonts.core.component import check_gpu_support
from gluonts.core.exception import GluonTSDataError
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import (
    Config as ForecastConfig,
    Forecast,
    SampleForecast,
)
from gluonts.model.predictor import Predictor
from gluonts.shell import PathsEnvironment

MB = 1024 * 1024
MODEL_SERVER_WORKERS = int(os.environ.get('MODEL_SERVER_WORKERS', -1))
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 6 * MB))

# inference environment variables set by SageMaker in batch mode
SAGEMAKER_BATCH = os.environ.get('SAGEMAKER_BATCH', 'false').lower() == 'true'
SAGEMAKER_MAX_PAYLOAD_IN_MB = int(os.getenv('SAGEMAKER_MAX_PAYLOAD_IN_MB', 6))
SAGEMAKER_MAX_CONCURRENT_TRANSFORMS = int(
    os.environ.get('SAGEMAKER_MAX_CONCURRENT_TRANSFORMS', 2 ** 32 - 1)
)


def number_of_workers(app: flask.Flask) -> int:
    logger = app.logger

    cpu_count = multiprocessing.cpu_count()

    if MODEL_SERVER_WORKERS > 0:
        try:
            logger.info(
                'Using {} workers (set by MODEL_SERVER_WORKERS environment '
                'variable).'.format(MODEL_SERVER_WORKERS)
            )
            return MODEL_SERVER_WORKERS
        except ValueError as ex:
            raise GluonTSDataError(
                'Cannot parse "inference worker count" '
                'parameter `{}` to int.'.format(ex)
            )

    elif SAGEMAKER_BATCH and SAGEMAKER_MAX_CONCURRENT_TRANSFORMS < cpu_count:
        logger.info(
            'Using {} workers (set by MaxConcurrentTransforms parameter in '
            'batch mode).'.format(SAGEMAKER_MAX_CONCURRENT_TRANSFORMS)
        )
        return SAGEMAKER_MAX_CONCURRENT_TRANSFORMS

    else:
        logger.info('Using {} workers'.format(cpu_count))
        return cpu_count


class DefaultShell(BaseApplication):
    DEFAULT_PORT = 8080

    def init(self, parser, opts, args) -> None:
        pass

    def __init__(
        self,
        paths: PathsEnvironment = PathsEnvironment(),
        port: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> None:
        app = flask.Flask('GluonTS scoring service')

        port = port if port else self.DEFAULT_PORT
        options = {
            "bind": f"0.0.0.0:{port}",
            "workers": workers if workers else number_of_workers(app),
            # "post_worker_init": ScoringService.post_worker_init,
            "timeout": 100,
        }

        check_gpu_support()

        predictor = Predictor.deserialize(paths.model)

        @app.route('/ping')
        def ping():
            return ''

        @app.route("/execution-parameters")
        def execution_parameters():
            return flask.jsonify(
                {
                    'MaxConcurrentTransforms': options['workers'],
                    'BatchStrategy': 'SINGLE_RECORD',
                    'MaxPayloadInMB': SAGEMAKER_MAX_PAYLOAD_IN_MB,
                }
            )

        @app.route('/invocations', methods=['POST'])
        def invocations() -> Any:
            try:
                payload = flask.request.json
                configuration = payload['configuration']
                if 'num_samples' in configuration:
                    configuration['num_eval_samples'] = configuration[
                        'num_samples'
                    ]
                config = ForecastConfig.parse_obj(configuration)

                def process(forecast: Forecast) -> dict:
                    prediction = {}
                    if 'samples' in config.output_types:
                        if isinstance(forecast, SampleForecast):
                            prediction['samples'] = forecast.samples.tolist()
                        else:
                            prediction['samples'] = []
                    if 'mean' in config.output_types:
                        prediction['mean'] = forecast.mean.tolist()
                    if 'quantiles' in config.output_types:
                        prediction['quantiles'] = {
                            q: forecast.quantile(q).tolist()
                            for q in config.quantiles
                        }
                    return prediction

                dataset = ListDataset(payload['instances'], predictor.freq)

                predictions = list(
                    map(
                        process,
                        predictor.predict(
                            dataset, no_samples=config.num_eval_samples
                        ),
                    )
                )
                return flask.jsonify(predictions=predictions)

            except Exception as error:
                return flask.jsonify(error=traceback.format_exc()), 500

        # NOTE: Stop Flask application when SIGTERM is received as a result
        # of "docker stop" command.
        signal.signal(signal.SIGTERM, self.stop)

        self.options = options
        self.application = app

        super(DefaultShell, self).__init__()

    def load_config(self) -> None:
        for key, value in self.options.items():
            key = key.lower()
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)

    def load(self) -> flask.Flask:
        return self.application

    def stop(self, *args, **kwargs):
        self.application.logger.info('Shutting down GluonTS scoring service')

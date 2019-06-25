# Standard library imports
import json
import logging
import os
from pydoc import locate
from typing import Callable, NamedTuple, Any, TypeVar, Type, Tuple

# First-party imports
from gluonts.core.component import check_gpu_support
from gluonts.core.exception import (
    GluonTSHyperparameterParseError,
    GluonTSFatalError,
    GluonTSException,
    GluonTSDataError,
)
from gluonts.core import log
from gluonts.dataset.common import Channel
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.evaluation import Evaluator, backtest
from gluonts.model.estimator import Estimator, GluonEstimator
from gluonts.model.predictor import Predictor, GluonPredictor
from gluonts.transform import FilterTransformation, TransformedDataset
from gluonts.shell import PathsEnvironment

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)


class DefaultShell(NamedTuple):
    paths: PathsEnvironment
    hyperparameters: dict
    from_hyperparameters: Callable[..., Estimator]

    def run(self):
        try:
            check_gpu_support()

            datasets = get_channels(self.paths).get_datasets()
            self.hyperparameters['freq'] = datasets.metadata.time_granularity

            estimator, predictor = self.run_train(datasets.train)

            if datasets.test is not None:
                self.run_test(datasets.test, estimator, predictor)

            predictor.serialize(self.paths.model)

        # TODO: do we want to handle GluonTS exceptions differently?
        except Exception as e:
            # logger.error(e)
            raise e

    def run_train(self, dataset) -> Tuple[Estimator, Predictor]:
        # print out stats
        log.metric('train_dataset_stats', dataset.calc_stats())

        estimator = self.from_hyperparameters(**self.hyperparameters)
        log.metric('estimator', estimator)

        return estimator, estimator.train(dataset)

    def run_test(self, dataset, estimator, predictor):
        test_dataset = TransformedDataset(
            dataset,
            transformations=[
                FilterTransformation(
                    lambda el: el['target'].shape[-1]
                    > predictor.prediction_length
                )
            ],
        )

        len_orig = len(dataset)
        len_filtered = len(test_dataset)
        if len_orig > len_filtered:
            logging.warning(
                'Not all time-series in the test-channel have '
                'enough data to be used for evaluation. Proceeding with '
                f'{len_filtered}/{len_orig} '
                f'(~{int(len_filtered/len_orig*100)}%) items.'
            )

        try:
            log.metric('test_dataset_stats', test_dataset.calc_stats())
        except GluonTSDataError as error:
            logging.error(
                f"Failure whilst calculating stats for test dataset: {error}"
            )
            return

        if isinstance(estimator, GluonEstimator) and isinstance(
            predictor, GluonPredictor
        ):
            inference_data_loader = InferenceDataLoader(
                dataset=test_dataset,
                transform=predictor.input_transform,
                batch_size=estimator.trainer.batch_size,
                ctx=estimator.trainer.ctx,
                float_type=estimator.float_type,
            )

            if estimator.trainer.hybridize:
                predictor.hybridize(batch=next(iter(inference_data_loader)))

            if self.hyperparameters.get('use_symbol_block_predictor'):
                predictor = predictor.as_symbol_block_predictor(
                    batch=next(iter(inference_data_loader))
                )

        num_eval_samples = self.hyperparameters.get('num_eval_samples', 100)
        quantiles = self.hyperparameters.get(
            'quantiles', (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        )

        # we only log aggregate metrics for now as item metrics may be
        # very large
        predictions, input_timeseries = backtest.make_evaluation_predictions(
            test_dataset, predictor, num_eval_samples
        )
        agg_metrics, _item_metrics = Evaluator(quantiles=quantiles)(
            input_timeseries, predictions, num_series=len_filtered
        )
        log.metric("agg_metrics", agg_metrics)


def get_hyperparameters(paths: PathsEnvironment) -> dict:
    with open(paths.config / 'hyperparameters.json') as json_file:
        # values passed through the SageMaker API are encoded as strings
        # pro-actively decode values that seem like arrays or objects
        raw_values = json.load(json_file)
        hyperparameters = {
            k: parse_value(k, v.strip() if type(v) == str else v)
            for k, v in raw_values.items()
        }
        logger.info(f'Available hyperparameters: {hyperparameters}')
        return hyperparameters


def get_channels(paths: PathsEnvironment) -> Channel:
    # FIXME: read inputdataconfig.json
    channel_names = os.listdir(str(paths.data))
    logger.info(f'Available channels: {channel_names}')
    # FIXME: catch and transform errors as CustomerError
    return Channel.parse_obj(
        {channel: paths.data / channel for channel in channel_names}
    )


def parse_value(k: str, v: Any) -> Any:
    # re-interpret as a list
    if type(v) == str and '[' == v[0] and v[-1] == ']':
        try:
            return json.loads(v)
        except ValueError as e:
            raise GluonTSHyperparameterParseError(k, v, 'list') from e
    # re-interpret as a dict
    elif type(v) == str and '{' == v[0] and v[-1] == '}':
        try:
            return json.loads(v)
        except ValueError as e:
            raise GluonTSHyperparameterParseError(k, v, 'dict') from e
    # do not re-interpret (Pydantic handles type conversions downstream
    else:
        return v


def get_estimator_path(hyperparameters: dict) -> str:
    if 'SWIST_ESTIMATOR_CLASS' in os.environ:
        logger.info(
            'Picking up GluonTS estimator classname from a '
            '"SWIST_ESTIMATOR_CLASS" environment variable.'
        )
        return os.environ['SWIST_ESTIMATOR_CLASS']

    elif 'estimator_class' in hyperparameters:
        logger.info(
            'Picking up GluonTS estimator classname from an "estimator_class" '
            'hyperparameter value.'
        )
        return hyperparameters['estimator_class']

    raise GluonTSFatalError(
        'Cannot determine the GluonTS estimator classname (missing variable '
        '"SWIST_ESTIMATOR_CLASS").'
    )


EstimatorT = TypeVar('EstimatorT', bound=Estimator)


def load_estimator(class_path: str) -> Type[EstimatorT]:
    estimator_class = locate(class_path)

    if estimator_class is None:
        raise GluonTSFatalError(
            f'Cannot locate estimator with classname "{class_path}".'
        )

    return estimator_class


def run() -> None:
    paths = PathsEnvironment()
    hyperparameters = get_hyperparameters(paths)

    estimator_path = get_estimator_path(hyperparameters)
    estimator_class = load_estimator(estimator_path)

    try:
        from_hyperparameters = getattr(estimator_class, "from_hyperparameters")
    except AttributeError:
        raise GluonTSFatalError(
            'Cannot find static method "from_hyperparameters" for estimator '
            f'"{estimator_path}"'
        )

    DefaultShell(paths, hyperparameters, from_hyperparameters).run()


def main() -> None:
    try:
        run()
    except GluonTSException as error:
        logging.error(error)
        exit(-1)


if __name__ == '__main__':
    main()

import logging
import multiprocessing


def number_of_workers(settings) -> int:
    cpu_count = multiprocessing.cpu_count()

    if settings.model_server_workers:
        logging.info(
            f'Using {settings.model_server_workers} workers '
            '(set by MODEL_SERVER_WORKERS environment variable).'
        )
        return settings.model_server_workers

    elif (
        settings.sagemaker_batch
        and settings.sagemaker_max_concurrent_transforms < cpu_count
    ):
        logging.info(
            f'Using {settings.sagemaker_max_concurrent_transforms} workers '
            '(set by MaxConcurrentTransforms parameter in batch mode).'
        )
        return settings.sagemaker_max_concurrent_transforms

    else:
        logging.info(f'Using {cpu_count} workers')
        return cpu_count

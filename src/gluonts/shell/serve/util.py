import logging
import multiprocessing


def number_of_workers(env) -> int:
    cpu_count = multiprocessing.cpu_count()

    if env.model_server_workers:
        logging.info(
            f'Using {env.model_server_workers} workers '
            '(set by MODEL_SERVER_WORKERS environment variable).'
        )
        return env.model_server_workers

    elif (
        env.sagemaker_batch
        and env.sagemaker_max_concurrent_transforms < cpu_count
    ):
        logging.info(
            f'Using {env.sagemaker_max_concurrent_transforms} workers '
            '(set by MaxConcurrentTransforms parameter in batch mode).'
        )
        return env.sagemaker_max_concurrent_transforms

    else:
        logging.info(f'Using {cpu_count} workers')
        return cpu_count

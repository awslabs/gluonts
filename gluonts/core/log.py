# Standard library imports
import logging
import os
from typing import Any

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

logger = logging.getLogger('SWIST')


def metric(metric: str, value: Any) -> None:
    logger.info(f'gluonts[{metric}]: {value!r}')

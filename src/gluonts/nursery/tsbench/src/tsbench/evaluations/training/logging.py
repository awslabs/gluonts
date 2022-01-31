import logging
from typing import Any
from gluonts.core.serde import dump_code

logger = logging.getLogger(__name__)


def log_metric(metric: str, value: Any) -> None:
    """
    Logs the provided metric in the format `gluonts[<metric>]: <value>`.
    """
    # pylint: disable=logging-fstring-interpolation
    logger.info(f"gluonts[{metric}]: {dump_code(value)}")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S %z]",
)


debug = logging.debug
info = logging.info
warn = logging.warn
error = logging.error

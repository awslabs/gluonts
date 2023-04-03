from pkgutil import extend_path

from pkg_resources import get_distribution, DistributionNotFound

from .trainer import Trainer, Trainer_adv

__path__ = extend_path(__path__, __name__)  # type: ignore

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "0.0.0-unknown"

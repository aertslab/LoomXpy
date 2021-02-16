
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution("loomxpy").version
except DistributionNotFound:
    pass
del DistributionNotFound, get_distribution

from .loomxpy import SCopeLoom


"""
Objects which define a specific domain to optimize over. These objects define
how to generate initial (policy agnostic) samples from the domain as well as to
optimize an index strategy over the domain.
"""

# pylint: disable=wildcard-import

from .domains import *
from . import domains

__all__ = []
__all__ += domains.__all__

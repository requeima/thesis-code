"""
Objects which implement the function interface.
"""

# pylint: disable=wildcard-import

from .basic import *

from . import basic

# import but don't add to __all__
from ._core import Mean

__all__ = []
__all__ += basic.__all__

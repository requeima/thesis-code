"""
Various utility functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = ['rstate', 'array2string', 'setter']


def rstate(rng=None):
    """
    Return a RandomState object. This is just a simple wrapper such that if rng
    is already an instance of RandomState it will be passed through, otherwise
    it will create a RandomState object using rng as its seed.
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    return rng


def array2string(a, sign=False):
    """
    Return a concise description of the array `a` as a string. This is not
    meant to be equivalent with `repr`, and should omit most of the data for
    large arrays.
    """
    a = np.array(a)
    f2s = '{: .3f}'.format if sign else '{:.3f}'.format
    if a.shape == ():
        string = f2s(a.flat[0])
    else:
        n = min(a.shape[-1], 2)
        fullrow = (n == a.shape[-1])
        string = '[' * a.ndim
        string += ', '.join(f2s(_) for _ in a.flat[:n])
        if fullrow:
            string += ']'
        if a.size > min(n, 3):
            string += ', ...'
        string += ']' * (a.ndim - fullrow)
    return string


def setter(func):
    """
    Decorator for a setter-only property.
    """
    return property(None, func, None, func.__doc__)



def create_getstate(slots):
    """
    Helper function to create the `__getstate__` method so that object can be pickled.
    """
    def getstate(self):
        return tuple(getattr(self, slt) for slt in slots)
    return getstate

def create_setstate(state):
    """
    Helper function to create the `__setstate__` method so that object can be pickled.
    """
    def setstate(self):
        for attr, val in zip(self, state):
            setattr(self, attr, val)

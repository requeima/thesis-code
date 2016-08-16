"""
Definition of a singleton-like class that allows memoization on its parameters.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import inspect
import weakref
from ..utils.misc import create_setstate, create_getstate


__all__ = ['Singleton', 'singleton']


def singleton(cls):
    """
    Decorator which takes a singleton class and replaces it by its instance.
    """
    return cls()


def int_(x):
    """
    Convert the given value `x` to an integer and raise an exception if any
    form of rounding occurs.
    """
    x_ = int(x)
    if x_ != x:
        raise ValueError('{} cannot be interpreted as an integer'.format(x))
    return x_


class Singleton(object):
    """
    Singleton class.
    """
    __slots__ = ['__weakref__', '__args']
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    __instances = weakref.WeakValueDictionary()

    def __new__(cls, *args, **kwargs):
        if hasattr(cls, '_init'):
            spec = inspect.getargspec(cls._init)
            narg = len(args)
            args = args + tuple(kwargs[_] for _ in spec.args[1+narg:])

            # do any type conversion
            if hasattr(cls, '__types__'):
                conv = dict(i=int_, f=float)
                args = tuple(conv[t](a) for (t, a) in zip(cls.__types__, args))

        else:
            if len(args) > 0 or len(kwargs) > 0:
                raise ValueError('constructor takes no arguments')
            args = ()

        # identify the instance with its class/arguments tuple
        inid = (cls, args)

        # construct the instance if necessary
        if inid in cls.__instances:
            instance = cls.__instances[inid]
        else:
            instance = super(Singleton, cls).__new__(cls, *args)
            if hasattr(cls, '_init'):
                instance._init(*args)   # pylint: disable=W0212
            instance.__args = args      # pylint: disable=W0212
            cls.__instances[inid] = instance

        return instance

    def __init__(self, *args, **kwargs):
        # the real initialization happens in _init so that we can make sure
        # that this is called only once for each singleton instance. this just
        # makes sure that __new__ can be called with any args/kwargs. any error
        # checking of the args should occur in _init as well.
        pass

    def __copy__(self):
        return self

    def __deepcopy__(self, _):
        return self

    def __repr__(self):
        string = self.__class__.__name__
        if self.__args != ():
            string += '('
            string += ', '.join('{!r}'.format(_) for _ in self.__args)
            string += ')'
        return string

"""
Tests for the singleton helper class.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import copy
import numpy.testing as nt

from reggie.core._singleton import Singleton, singleton


class Singleton1(Singleton):
    pass


class Singleton2(Singleton):
    __types__ = 'if'

    def _init(self, a, b):
        pass


class Singleton3(Singleton):
    def _init(self, a):
        pass


def test_singleton():
    Singleton3(1)
    Singleton3(3)

    instance = Singleton2(1, 1.2)

    assert instance is Singleton2(1, 1.2)
    assert instance is not Singleton2(1, 1.0)
    assert instance is copy.copy(instance)
    assert instance is copy.deepcopy(instance)
    assert Singleton1() is singleton(Singleton1)
    assert isinstance(repr(instance), str)
    assert isinstance(repr(Singleton1()), str)

    nt.assert_raises(ValueError, Singleton1, 1)
    nt.assert_raises(ValueError, Singleton2, 2.3, 2.4)

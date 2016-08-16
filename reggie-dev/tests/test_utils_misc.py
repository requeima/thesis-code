"""
Tests for misc utilities.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt

from reggie.utils.misc import rstate, array2string, setter


def test_rstate():
    rng1 = rstate()
    rng2 = rstate(1)
    rng3 = rstate(rng2)

    nt.assert_(isinstance(rng1, np.random.RandomState))
    nt.assert_(isinstance(rng2, np.random.RandomState))
    nt.assert_(isinstance(rng3, np.random.RandomState))
    nt.assert_(rng3 is rng2)


def test_array2string():
    nt.assert_(array2string(1) == '1.000')
    nt.assert_(array2string([[1, 2]]) == '[[1.000, 2.000]]')
    nt.assert_(array2string([1, 2, 3]) == '[1.000, 2.000, ...]')
    nt.assert_(array2string([[1], [2]]) == '[[1.000], ...]')


class Foo(object):
    @setter
    def foo(self, val):
        "foo"
        pass


def test_setter():
    foo = Foo()
    foo.foo = 1
    nt.assert_raises(AttributeError, getattr, foo, 'foo')

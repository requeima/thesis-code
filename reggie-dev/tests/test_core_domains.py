"""
Tests for the domain instances.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

from reggie.core.domains import Real, Positive, Bounded
from reggie.core.domains import LogTransform, epsilon


def test_domains():
    # test inits
    nt.assert_raises(ValueError, Bounded, 1, -1)

    # test subsets
    nt.assert_(Bounded(1, 5) <= Positive())
    nt.assert_(Bounded(-1, 5) <= Real())
    nt.assert_(Real() <= Real())
    nt.assert_(Positive() <= Real())
    nt.assert_(not Bounded(-1, 5) <= Positive())

    # test containment
    nt.assert_(1.0 in Real())
    nt.assert_(1.0 in Positive())
    nt.assert_(1.0 in Bounded(0, 1))
    nt.assert_(-1.0 not in Bounded(0, 1))
    nt.assert_(-1.0 not in Positive())

    nt.assert_(Real().project(5) == 5)
    nt.assert_(Positive().project(-1) == epsilon)
    nt.assert_(Bounded(0, 1).project(-1) == 0)


def test_logtransform():
    inverse = LogTransform().get_inverse(LogTransform().get_transform(0.1))
    nt.assert_allclose(inverse, 0.1)
    assert LogTransform().get_inverse(-np.inf) == epsilon


def test_logtransform_image():
    assert LogTransform().get_image(Positive()) == Real()
    assert LogTransform().get_image(Bounded(1, 2)) is Bounded(0, np.log(2))
    nt.assert_raises(ValueError, LogTransform().get_image, Real())


def test_logtransform_gradient():
    for t in [0.1, 0.5, 1, 2]:
        t = np.array([t])
        x = LogTransform().get_inverse(t)
        d1 = LogTransform().get_gradfactor(x)
        d2 = spop.approx_fprime(t, LogTransform().get_inverse, 1e-8)
        nt.assert_allclose(d1, d2)

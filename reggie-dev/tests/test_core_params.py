"""
Tests for the the core parameterization objects.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt

from reggie.core.params import Parameterized
from reggie.core.domains import Real, Positive
from reggie.core.priors import Uniform


class Kernel(Parameterized):
    pass


class SEARD(Kernel):
    def __init__(self, rho, ell):
        super(SEARD, self).__init__(
            ('rho', rho, Positive()),
            ('ell', ell, Positive(), 'd'))


class GP(Parameterized):
    def __init__(self, kern, bias, infer='exact'):
        super(GP, self).__init__(
            ('kern', kern, Kernel),
            ('bias', bias, Real()), infer=infer)


class Model(Parameterized):
    def __init__(self, submodel):
        super(Model, self).__init__(('submodel', submodel))


def test_parameterized():
    nt.assert_raises(ValueError, SEARD, 3, 'asdf')
    nt.assert_raises(ValueError, SEARD, 3, np.random.rand(3, 3))
    nt.assert_raises(ValueError, SEARD, 3, -1)
    nt.assert_raises(ValueError, GP, Model(SEARD(1, 1)), 1.0)

    kern = SEARD(3, 3)
    gp1 = GP(kern, 3)
    gp2 = gp1.copy()

    gp1.params['kern'].block = 1
    blocks = set(map(tuple, gp1.hyper_blocks))
    nt.assert_((2,) in blocks)
    nt.assert_((0, 1) in blocks)
    nt.assert_raises(ValueError, setattr, gp1.params, 'block', 'asdf')

    gp1.params.describe()
    gp1.hyper = np.array([0, 0, 1])
    gp1.params.prior = Uniform(0.1, 10)
    gp2.params.value = np.array([2, 2, 2])

    # this assumes the hypers are [0, 0, 1]
    nt.assert_equal(gp1.get_logjacobian(), 0)

    nt.assert_equal(gp2._kern._rho, np.array(2.0))
    nt.assert_equal(gp2._kern._ell, np.array([2.0]))

    nt.assert_(isinstance(repr(gp1), str))

    nt.assert_raises(IndexError, gp1.params['kern']['ell'][0].__getitem__, 0)
    nt.assert_raises(IndexError, gp1.params['kern']['ell'].__getitem__, 'asdf')
    nt.assert_raises(IndexError, gp1.params['kern']['ell'].__getitem__, (0, 0))
    nt.assert_raises(KeyError, gp1.params.__getitem__, 'asdf')
    nt.assert_raises(ValueError, setattr, gp1.params, 'value', np.r_[-1, 1, 1])
    nt.assert_raises(ValueError, setattr, gp1.params, 'prior', Uniform(-1, 1))
    nt.assert_raises(ValueError, setattr, gp1.params, 'prior', 'asdf')

    nt.assert_equal(gp1.params['kern'].value, np.r_[1, 1.])
    nt.assert_equal(gp1.params['kern']['ell'][0].value, np.r_[1.])
    nt.assert_equal(gp1.params.value, np.r_[1, 1, 1.])
    nt.assert_equal(gp2.params.value, np.r_[2, 2, 2.])
    nt.assert_equal(gp2.hyper, np.r_[np.log(2), np.log(2), 2.])
    nt.assert_equal(gp2.hyper_bounds, [(None, None)] * 3)

    nt.assert_equal(gp2._wrap_gradient(np.r_[2, 2, 2.]), np.r_[4, 4, 2.])
    nt.assert_equal(gp2._kern._wrap_gradient(np.r_[2, 2.]), np.r_[2, 2.])

    nt.assert_equal(gp2.get_logprior(), 0)
    nt.assert_equal(gp2.get_logprior(True), (0, np.r_[0, 0, 0.]))

    nt.assert_raises(AttributeError, getattr, gp1._kern, 'hyper')
    nt.assert_raises(AttributeError, getattr, gp1._kern, 'hyper_bounds')
    nt.assert_raises(AttributeError, gp1._kern.get_logprior)
    nt.assert_raises(AttributeError, gp1._kern.get_logjacobian)
    nt.assert_raises(AttributeError, setattr, gp1._kern, 'hyper', 1)

    nt.assert_warns(UserWarning, setattr, gp1.params, 'prior', Uniform(10, 11))

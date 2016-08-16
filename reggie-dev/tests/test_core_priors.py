"""
Tests for priors.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import nose
import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

from reggie.core.domains import Domain
from reggie.core.priors import Uniform, Normal, LogNormal, Horseshoe


# BASE TEST CLASS #############################################################

class PriorTest(object):
    def __init__(self, prior):
        self.prior = prior

    def test_repr(self):
        _ = repr(self.prior)

    def test_domain(self):
        assert isinstance(self.prior.domain, Domain)

    def test_sample(self):
        try:
            assert self.prior.sample(10, 0).shape == (10,)
        except NotImplementedError:
            raise nose.SkipTest()

    def test_logprior(self):
        for t in [1, 2, 3]:
            t = np.array([t])
            g1 = spop.approx_fprime(t, self.prior.get_logprior, 1e-8)
            _, g2 = self.prior.get_logprior(t, True)
            nt.assert_allclose(g1, g2, rtol=1e-6)


# PER-INSTANCE TESTS ##########################################################

class TestUniform(PriorTest):
    def __init__(self):
        PriorTest.__init__(self, Uniform(0, 10))


class TestNormal(PriorTest):
    def __init__(self):
        PriorTest.__init__(self, Normal(0, 1))


class TestLogNormal(PriorTest):
    def __init__(self):
        PriorTest.__init__(self, LogNormal(0, 1))


class TestHorseshoe(PriorTest):
    def __init__(self):
        PriorTest.__init__(self, Horseshoe(1))


# INITIALIZATION TESTS ########################################################

def test_uniform_init():
    nt.assert_raises(ValueError, Uniform, 0, -1)

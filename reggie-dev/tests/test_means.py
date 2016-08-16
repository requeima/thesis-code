"""
Tests for mean objects.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

import reggie.means as means


# BASE TEST CLASSES ###########################################################

class MeanTest(object):
    def __init__(self, mean, X):
        self.mean = mean
        self.X = X

    def test_repr(self):
        _ = repr(self.mean)

    def test_call(self):
        F1 = self.mean.get_mean(self.X)
        F2 = np.array([self.mean(x) for x in self.X])
        nt.assert_equal(F1, F2)

    def test_get_grad(self):
        if self.mean.hyper.size > 0:
            f = lambda hyper, x: self.mean.copy(hyper)(x)
            G1 = self.mean.get_grad(self.X)
            G2 = np.array([spop.approx_fprime(self.mean.hyper, f, 1e-8, x_)
                           for x_ in self.X]).T
            nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)
        else:
            assert self.mean.get_grad(self.X).shape == (0, len(self.X))


class RealMeanTest(MeanTest):
    def __init__(self, mean):
        rng = np.random.RandomState(0)
        X = rng.rand(5, mean.ndim)
        super(RealMeanTest, self).__init__(mean, X)

    def test_get_gradx(self):
        mu = self.mean
        G1 = self.mean.get_gradx(self.X)
        G2 = np.array([spop.approx_fprime(x, mu, 1e-8) for x in self.X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)


# PER-INSTANCE TEST CLASSES ###################################################

class TestZero(MeanTest):
    def __init__(self):
        f = means.Zero()
        X = np.random.rand(5, 2)
        MeanTest.__init__(self, f, X)

    def test_get_gradx(self):
        nt.assert_equal(self.mean.get_gradx(self.X), np.zeros_like(self.X))


class TestConstant(MeanTest):
    def __init__(self):
        f = means.Constant(1.0)
        X = np.random.rand(5, 2)
        MeanTest.__init__(self, f, X)

    def test_get_gradx(self):
        nt.assert_equal(self.mean.get_gradx(self.X), np.zeros_like(self.X))


class TestLinear(RealMeanTest):
    def __init__(self):
        RealMeanTest.__init__(self, means.Linear([2, 3], 3))

"""
Tests for models.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

from reggie import make_gp


# BASE TEST CLASSES ###########################################################

class ModelTest(object):
    def __init__(self, model, X, Y):
        self.X = X
        self.Y = Y
        self.model0 = model.copy()
        self.model = model
        self.model.add_data(X, Y)

    def test_repr(self):
        _ = repr(self.model)

    def test_add_data(self):
        n = len(self.X) // 2
        self.model0.add_data(self.X[:n], self.Y[:n])
        self.model0.add_data(self.X[n:], self.Y[n:])
        nt.assert_allclose(self.model.predict(self.X),
                           self.model0.predict(self.X))

    def test_get_loglike(self):
        # make sure we can evaluate the log-likelihood with/without data
        for model in [self.model, self.model0]:
            _ = model.get_loglike()
            _, _ = model.get_loglike(True)

        # the function we'll take the derivatives of
        f = lambda hyper: self.model.copy(hyper).get_loglike()

        # construct the derivatives to compare
        _, g1 = self.model.get_loglike(grad=True)
        g2 = spop.approx_fprime(self.model.hyper, f, 1e-8)

        # must be close
        nt.assert_allclose(g1, g2, rtol=1e-6, atol=1e-6)

    """
    def test_predict(self):
        # first check that we can even evaluate the posterior.
        _ = self.model.predict(self.X)

        # check the mu gradients
        mu = lambda x: self.model.predict(x[None])[0]
        s2 = lambda x: self.model.predict(x[None])[1]

        # check the mu derivatives wrt x.
        G1 = self.model.predict(self.X, grad=True)[2]
        G2 = np.array([spop.approx_fprime(x, mu, 1e-8) for x in self.X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

        # check the s2 derivatives wrt x
        G1 = self.model.predict(self.X, grad=True)[3]
        G2 = np.array([spop.approx_fprime(x, s2, 1e-8) for x in self.X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)
    """


# PER-INSTANCE TEST CLASSES ###################################################

class TestGP(ModelTest):
    def __init__(self):
        X = np.random.rand(10, 2)
        Y = np.random.rand(10)
        gp = make_gp(1, 1, [1., 1.])
        ModelTest.__init__(self, gp, X, Y)

    def test_fstar(self):
        # first check that we can even evaluate the posterior.
        X = self.model._X
        model = self.model.condition_fstar(0.7)

        # check the mu gradients
        f = lambda x: model.predict(x[None], True)[0]
        G1 = model.predict(X, True)[2]
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

        # check the s2 gradients
        f = lambda x: model.predict(x[None])[1]
        G1 = model.predict(X, grad=True)[3]
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

        # check the s2 gradients
        f = lambda x: model.get_entropy(x[None])
        G1 = model.get_entropy(X, grad=True)[1]
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)


class TestGP_FITC(ModelTest):
    def __init__(self):
        X = np.random.rand(10, 2)
        Y = np.random.rand(10)
        U = np.random.rand(50, 2)
        gp = make_gp(0.7, 1, [1., 1.], inf='fitc', U=U)
        ModelTest.__init__(self, gp, X, Y)


class TestGP_Laplace(ModelTest):
    def __init__(self):
        X = np.random.rand(10, 2)
        Y = np.random.rand(10)
        gp = make_gp(0.7, 1, [1., 1.], inf='laplace')
        ModelTest.__init__(self, gp, X, Y)

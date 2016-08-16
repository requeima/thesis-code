"""
Tests for kernel objects.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop
import nose

import reggie.kernels as kernels


# BASE TEST CLASSES ###########################################################

class KernelTest(object):
    def __init__(self, kernel, X1, X2):
        self.kernel = kernel
        self.X1 = X1
        self.X2 = X2

    def test_repr(self):
        _ = repr(self.kernel)

    def test_call(self):
        m = self.X1.shape[0]
        n = self.X2.shape[0]
        K = self.kernel.get_kernel(self.X1, self.X2)
        K_ = np.array([self.kernel(x1, x2)
                       for x1 in self.X1
                       for x2 in self.X2]).reshape(m, n)
        nt.assert_equal(K, K_)

    def test_get_kernel(self):
        K = self.kernel.get_kernel(self.X1, self.X1)
        k = self.kernel.get_dkernel(self.X1)
        nt.assert_allclose(k, K.diagonal())

    def test_get_grad(self):
        k = lambda hyper, x1, x2: self.kernel.copy(hyper)(x1, x2)
        x = self.kernel.hyper
        G = self.kernel.get_grad(self.X1, self.X2)
        m = self.X1.shape[0]
        n = self.X2.shape[0]
        G_ = np.array([spop.approx_fprime(x, k, 1e-8, x1, x2)
                       for x1 in self.X1
                       for x2 in self.X2]).swapaxes(0, 1).reshape(-1, m, n)

        for i in range(0, G.shape[0]):
            for j in range(G.shape[1]):
                print(G_[i,j, :])
                print(G[i, j, :])
                print(" ")

        nt.assert_allclose(G, G_, rtol=1e-6, atol=1e-6)

    def test_get_dgrad(self):
        g = self.kernel.get_dgrad(self.X1)
        G = np.vstack(map(np.diag, self.kernel.get_grad(self.X1)))
        nt.assert_allclose(g, G)


class RealKernelTest(KernelTest):
    def __init__(self, kernel):
        rng = np.random.RandomState(0)
        if hasattr(kernel, 'warped'):
            X1 = np.empty((5, kernel.ndim))
            X2 = np.empty((3, kernel.ndim))
            for n in range(kernel.ndim):
                a = np.min(kernel.bounds[n, 0])
                b = np.max(kernel.bounds[n, 1])
                X1[:, n] = np.random.uniform(low=a, high=b, size=(5))
                X2[:, n] = np.random.uniform(low=a, high=b, size=(3))
        else:
            X1 = rng.rand(5, kernel.ndim)
            X2 = rng.rand(3, kernel.ndim)
        super(RealKernelTest, self).__init__(kernel, X1, X2)

    def test_get_gradx(self):
        G1 = self.kernel.get_gradx(self.X1, self.X2)
        m = self.X1.shape[0]
        n = self.X2.shape[0]
        d = self.X1.shape[1]
        k = self.kernel

        G2 = np.array([spop.approx_fprime(x1, k, 1e-8, x2)
                       for x1 in self.X1
                       for x2 in self.X2]).reshape(m, n, d)

        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_get_dgradx(self):
        g = self.kernel.get_dgradx(self.X1)
        G = self.kernel.get_gradx(self.X1)
        G = np.vstack(G[i, i] for i in xrange(len(G)))

        for i in range(G.shape[0]):
                print(g[i,:])
                print(G[i,:])

        nt.assert_allclose(g, G)

    def test_gradxy(self):
        try:
            G1 = self.kernel.get_gradxy(self.X1, self.X2)
        except NotImplementedError:
            raise nose.SkipTest()

        m = self.X1.shape[0]
        n = self.X2.shape[0]
        d = self.X1.shape[1]
        g = lambda x2, x1, i: self.kernel.get_gradx(x1[None],
                                                    x2[None])[0, 0, i]

        G2 = np.array([spop.approx_fprime(x2, g, 1e-8, x1, i)
                       for x1 in self.X1
                       for x2 in self.X2
                       for i in xrange(d)]).reshape(m, n, d, d)

        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_sample_spectrum(self):
        try:
            W, alpha = self.kernel.sample_spectrum(100)
        except NotImplementedError:
            raise nose.SkipTest()
        assert W.shape == (100, self.kernel.ndim)
        assert isinstance(alpha, float)


# INIT TESTS ##################################################################

def test_matern_init():
    nt.assert_raises(ValueError, kernels.MaternARD, 1, 1, 2)


# PER-INSTANCE TESTS ##########################################################

class TestSEARD(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.SEARD(0.8, [0.3, 0.4]))

class TestWARPSEARD(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.warp_se.WARPSEARD (0.8, [0.3, 0.6], [0.3, 0.1], [0.1, 0.2], [[0., 5.], [0., 0.5]]))

class TestPeriodic(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.Periodic(0.8, 0.6, 0.3))

class TestSpectralMixture(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.sm.SM([0.3, 0.7], [[1.5, -3.3], [-2.0, 1.1]], [[2.0, 1.0], [1.0, 1.0]]))

class TestPeriodicLinear(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.periodic_linear.PeriodicLinear(0.8, 0.2, 0.3, 2., 1.0, 5.0))


class TestMaternARD1(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.MaternARD(0.8, [0.3, 0.4], d=1))


class TestMaternARD3(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.MaternARD(0.8, [0.3, 0.4], d=3))


class TestMaternARD5(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, kernels.MaternARD(0.8, [0.3, 0.4], d=5))

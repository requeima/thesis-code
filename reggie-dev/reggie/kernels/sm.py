"""
Implementation of the spectral mixture kernel.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.domains import Positive, Real

from ._core import RealKernel
from ._distances import diff

__all__ = ['SM']


class SM(RealKernel):
    """
    The spectral mixture kernel with weights w, mean array Mu and diagonals Md. Mean and Variance given as matrices
    """
    def __init__(self, weights, Mu, Md):
        init_args = []

        for q in range(len(Mu)):
            init_args += [('weight_' + str(q), weights[q], Positive()),
                          ('Mu_' + str(q), Mu[q], Real(), 'd'),
                          ('Md_' + str(q), Md[q], Positive(), 'd')]
        super(SM, self).__init__(*init_args)
        # save the input dimension and the number of mixture components (weights)
        self.ndim = len(Md[q])
        self.nweights = len(Md)

    def _sum_weights(self):
        return np.sum([getattr(self, '_weight_' + str(q)) for q in range(self.nweights)])

    def get_component_params(self, q):
        # grab the weight and sufficient statistics for one component.
        w_q = getattr(self, '_weight_' + str(q))
        u_q = getattr(self, '_Mu_' + str(q))
        v_q = getattr(self, '_Md_' + str(q))
        return w_q, u_q, v_q

    def get_kernel(self, X1, X2=None):

        D = diff(X1, X2)
        for q in range(self.nweights):
            # grab the weight and sufficient statistics for one component
            w_q, u_q, v_q = self.get_component_params(q)

            # compute the kernel
            E = (np.exp(-2.*(np.pi**2)*np.square(D)*v_q[None, None, :])*
                 np.cos(2.*np.pi*D*u_q[None, None, :]))
            if 'k' not in locals():
                k = w_q*np.prod(E, axis=2)
            else:
                k += w_q*np.prod(E, axis=2)
        return k

    def get_dkernel(self, X1):
        # sum_weights = np.sum([getattr(self, '_weight_' + str(q)) for q in range(self.nweights)])
        return np.full(len(X1), 1.0)

    def _grad_components(self, D, q, n):
        w_q, u_q, v_q = self.get_component_params(q)
        E = np.exp(-2. * (np.pi ** 2) * np.square(D) * v_q[None, None, :]) * np.cos(2. * np.pi * D * u_q[None, None, :])
        E_minus = np.prod(E[:, :, :n], axis=2) * np.prod(E[:, :, n + 1:], axis=2)
        exp_arg = -2. * (np.pi**2) * np.square(D[:, :, n]) * v_q[n]
        trig_arg = 2. * np.pi * D[:, :, n] * u_q[n]
        return E, E_minus, exp_arg, trig_arg


    def get_grad(self, X1, X2=None):
        D = diff(X1, X2)
        G = np.empty((self.nweights*(2*self.ndim + 1), ) + (D.shape[0], D.shape[1]))
        count = 0
        for q in range(self.nweights):
            # grab the weight and sufficient statistics for one component
            w_q, u_q, v_q = self.get_component_params(q)
            # weight gradient
            E =  (np.exp(-2. * (np.pi ** 2) * np.square(D) * v_q[None, None, :]) *
                                              np.cos(2.*np.pi*D*u_q[None, None, :]))
            G[count] =  np.prod(E, axis=2)/self._sum_weights()
            count += 1

            # mean hyperparameter gradients
            for n in range(self.ndim):
                _, E_minus, exp_arg, trig_arg = self._grad_components(D, q, n)
                G[count] = -w_q * E_minus * np.exp(exp_arg) * np.sin(trig_arg) * 2. * np.pi * D[:, :, n]
                count += 1

            # variance hyperparameter gradients
            for n in range(self.ndim):
                _, E_minus, exp_arg, trig_arg = self._grad_components(D, q, n)
                G[count] = - (w_q * E_minus * 2. * (np.pi ** 2) * (D[:, :, n] ** 2) *
                              np.cos(trig_arg) * np.exp(exp_arg))

                count += 1

        return self._wrap_gradient(G)

    def get_dgrad(self, X1):
        G = np.zeros((self.nweights*(2*self.ndim + 1), X1.shape[0]))
        G[0, :] = 1.
        G[self.nweights*self.ndim + 1, :] = 1.
        return self._wrap_gradient(G)


    def get_gradx(self, X1, X2=None):
        D = diff(X1, X2)
        G = np.empty(D.shape)

        for n in range(self.ndim):
            g = np.zeros((D.shape[0], D.shape[1]))
            for q in range(self.nweights):
                w_q, u_q, v_q = self.get_component_params(q)
                _, E_minus, exp_arg, trig_arg = self._grad_components(D, q, n)
                A = np.exp(exp_arg)
                B = -4. * (np.pi ** 2) * D[:, :, n] * v_q[n] * np.cos(trig_arg)
                C = -2. * np.pi * u_q[n] * np.sin(trig_arg)
                g += w_q*E_minus*A*(B + C)
            G[:,:, n] = g

        return G

    def get_dgradx(self, X1):
        return np.zeros_like(X1)


    def get_gradxy(self, X1, X2=None):
        D = diff(X1, X2)
        G = np.empty(D.shape + (D.shape[2],))

        for n1 in range(self.ndim):
            for n2 in range(self.ndim):
                g = np.zeros((D.shape[0], D.shape[1]))
                for q in range(self.nweights):
                    w_q, u_q, v_q = self.get_component_params(q)
                    E, E_minus1, exp_arg1, trig_arg1 = self._grad_components(D, q, n1)
                    A1 = np.exp(exp_arg1)
                    B1 = -4. * (np.pi ** 2) * D[:, :, n1] * v_q[n1] * np.cos(trig_arg1)
                    C1 = -2. * np.pi * u_q[n1] * np.sin(trig_arg1)

                    if n1 != n2:
                        _, _, exp_arg2, trig_arg2 = self._grad_components(D, q, n2)
                        A2 = np.exp(exp_arg2)
                        B2 = -4. * (np.pi ** 2) * D[:, :, n2] * v_q[n2] * np.cos(trig_arg2)
                        C2 = -2. * np.pi * u_q[n2]* np.sin(trig_arg2)

                        n_max = np.max([n1, n2])
                        n_min = np.min([n1, n2])
                        E_double_minus = np.prod(E[:, :, :n_min], axis=2) * np.prod(E[:, :, n_min + 1:n_max], axis=2) * \
                                         np.prod(E[:, :, n_max + 1:], axis=2)

                        g += w_q * E_double_minus * (A1 * (B1 + C1)) * (A2 * (-B2 - C2))

                    if n1 == n2:
                        dA1 = 4.0 * (np.pi**2) * D[:,:,n1] * v_q[n1] * np.exp(exp_arg1)
                        dB1 = 4.0 * (np.pi**2) * v_q[n1] * (np.cos(trig_arg1) - trig_arg1 * np.sin(trig_arg1))
                        dC1 = 4.0 * ((np.pi * u_q[n1])**2) * np.cos(trig_arg1)
                        g += w_q * E_minus1 * ((dA1*(B1 + C1)) +  (A1*(dB1 + dC1)))

                G[:,:,n1,n2] = g

        return G


    def sample_spectrum(self, N, rng=None):
        # grab the normalized weights /2
        weights = 2*[self.get_component_params(q)[0] / (2 * self._sum_weights()) for q in range(self.nweights)]
        # how many of each gaussian to use
        gauss_use = np.random.multinomial(N, weights)

        W = np.empty((N, self.ndim))
        count = 0
        for q in range(self.nweights):
            _, u_q, v_q = self.get_component_params(q)
            n1 = gauss_use[q]
            n2 = gauss_use[self.nweights + q]
            W[count:count + n1] = u_q + np.random.multivariate_normal(np.zeros_like(u_q), np.diag(v_q), n1)
            count += n1
            W[count:count + n2] = -u_q + np.random.multivariate_normal(np.zeros_like(u_q), np.diag(v_q), n2)
            count  += n2

        return 2*np.pi*W, self._sum_weights()

"""
Implementation of the Gaussian likelihood.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss

from ..core.domains import Positive
from ..utils.misc import rstate
from ._core import Likelihood

__all__ = ['Gaussian']


class Gaussian(Likelihood):
    """
    The Gaussian likelihood function for regression. This likelihood can be
    written as::

        p(y|f) = 1 / sqrt(2*pi*sn2) * exp(-0.5 * (y-f)^2 / sn2)

    where `sn2` is the noise variance.
    """
    def __init__(self, sn2):
        super(Gaussian, self).__init__(('sn2', sn2, Positive()))

    def get_variance(self):
        return float(self._sn2)

    def predict(self, mu, s2):
        return mu, s2 + self._sn2

    def sample(self, f, rng=None):
        rng = rstate(rng)
        return f + rng.normal(size=len(f), scale=np.sqrt(self._sn2))

    def get_loglike(self, y, f):
        r = y-f
        lp = -0.5 * (r**2 / self._sn2 + np.log(2 * np.pi * self._sn2))
        d1 = r / self._sn2
        d2 = np.full_like(r, -1/self._sn2)
        d3 = np.zeros_like(r)
        return lp, d1, d2, d3

    def get_laplace_grad(self, y, f):
        r = y-f
        s = self._sn2**2
        d0 = 0.5 * (r**2/s - 1/self._sn2)
        d1 = -r/s
        d2 = np.full_like(r, 1/s)
        # there is only one parameter. So this makes it such that the gradient
        # is [(d0, d1, d2)], where each of the components are 1d vectors.
        return self._wrap_gradient(np.c_[d0, d1, d2].T[None])

    def get_tail(self, f, mu, s2, dmu=None, ds2=None):
        # standardize the variables and compute the CDF.
        a = mu - f
        s = np.sqrt(s2)
        z = a / s
        cdf = ss.norm.cdf(z)

        # just return PI if no gradient information is given; note also that
        # in this case PI is just given by the CDF.
        if dmu is None:
            return cdf

        # compute the gradient of PI (ie the cdf)
        dcdf = dmu / s[:, None] - 0.5 * ds2 * z[:, None] / s2[:, None]
        return cdf, dcdf

    def get_improvement(self, f, mu, s2, dmu=None, ds2=None):
        # get the posterior (possibly with gradients) and standardize
        a = mu - f
        s = np.sqrt(s2)
        z = a / s

        # get the cdf, pdf, integrate it to compute EI.
        cdf = ss.norm.cdf(z)
        pdf = ss.norm.pdf(z)
        ei = a * cdf + s * pdf

        # just return EI if no gradient information is given
        if dmu is None:
            return ei

        # compute the gradient of EI
        dei = 0.5 * ds2 / s2[:, None]
        dei *= (ei - s * z * cdf)[:, None] + cdf[:, None] * dmu
        return ei, dei

    def get_entropy(self, mu, s2, dmu=None, ds2=None):
        sp2 = s2 + self._sn2
        H = 0.5 * np.log(2 * np.pi * np.e * sp2)

        # just return the marginal entropy
        if dmu is None:
            return H

        # compute the gradient of the entropy
        dH = 0.5 * ds2 / sp2[:, None]
        return H, dH


"""
Implementation of the probit likelihood.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.special as ss

from ._core import Likelihood
from ..utils.misc import rstate

__all__ = ['Probit']


def logphi(z):
    lp = np.log(ss.erfc(-z/np.sqrt(2))/2)
    d1 = np.exp(-z**2/2 - lp) / np.sqrt(2*np.pi)
    d2 = -d1 * np.abs(z+d1)
    d3 = -d2 * np.abs(z+2*d1) - d1
    return lp, d1, d2, d3


class Probit(Likelihood):
    def __init__(self):
        super(Probit, self).__init__()

    def predict(self, mu, s2):
        return ss.ndtr(mu / np.sqrt(1+s2))

    def sample(self, f, rng=None):
        rng = rstate(rng)
        i = rng.rand(len(f)) < ss.ndtr(f)
        y = 1*i - 1*(~i)
        return y

    def get_loglike(self, y, f):
        # make sure we have only +/- signals
        i = (y == 1)
        y = 1*i - 1*(-i)
        lp, d1, d2, d3 = logphi(y*f)
        d1 *= y
        d3 *= y
        return lp, d1, d2, d3

    def get_laplace_grad(self, y, f):
        return np.zeros((0, 3, len(y)))

    def get_tail(self, f, mu, s2, dmu=None, ds2=None):
        # in the case of a probit f should represent a vector of probabilities,
        # i.e. each entry should be in (0, 1). ndtri is the inverse standard
        # normal CDF (inverse of the probit) so this transforms f into a vector
        # of real-values which we then evaluate ndtr on (the standard normal
        # CDF or probit) after normalizing by mu/s2.

        a = ss.ndtri(f)                     # inverse CDF of target
        z = (mu - a) / np.sqrt(1 + s2)      # standardize target
        p = ss.ndtr(z)                      # CDF

        if dmu is None:
            return p

        raise NotImplementedError

    def get_entropy(self, mu, s2, dmu=None, ds2=None):
        # run predictions to get the probability of being in class 1 and then
        # use that to compute the entropy in the standard way.
        p = self.predict(mu, s2)
        H = -p*np.log(p) - (1-p)*np.log(1-p)

        # just return the marginal entropy
        if dmu is None:
            return H

        raise NotImplementedError

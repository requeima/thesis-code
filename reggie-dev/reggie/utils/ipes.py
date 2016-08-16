"""
Utility functions for ipes acquisition
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# import numpy as np
import autograd.numpy as np
from autograd import grad as diff
from scipy.integrate import quad
from scipy.stats import norm


def gmm_entropy_integrand(y, mu, s2):
    """
    Helper function - returns the density of a gmm with means in vector
    mu and variances in vector s2 at the point y
    """
    #log-sum-exp trick for computing log of the gmm pdf
    log_gmm =  np.array(norm.logpdf(y, loc=mu, scale=np.sqrt(s2)))
    max_log = np.max(log_gmm)
    sum_log = np.sum(np.exp(log_gmm - max_log))
    log_p = max_log + np.log(sum_log) - np.log(len(log_gmm)) # subtract dividing constant in mean
    p = np.exp(log_p)
    # if p < 1e-10:
    #     return 0
    # else:
    return -p*log_p


def _get_ipes_entropy(X, models):
    "models is an iterable list of models"
    n = len(X)
    parts = np.array([_.predict(X) for _ in models])
    mu = parts[:, 0, :]  # dimensions are models x data
    s2 = parts[:, 1, :]  # dimensions are models x data
    return np.array([quad(gmm_entropy_integrand, -np.inf, np.inf, args=(mu[:, i], s2[:, i]))[0] for i in range(n)])


def get_ipes_entropy(X, models, grad=False):
    H = _get_ipes_entropy(X, models)
    if grad is False:
        return H
    else:
        dH = diff(_get_ipes_entropy)
        H, dH(X, meta_model)
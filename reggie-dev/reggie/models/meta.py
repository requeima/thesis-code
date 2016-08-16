"""
Meta-models for learning.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm



from ._core import Model
from ..learning import sample
from ..utils.misc import rstate


__all__ = ['MCMC']


def _integrate(parts, grad):
    """
    Helper function to integrate over a function and potentially its
    derivatives.
    """
    if grad:
        return tuple([np.mean(_, axis=0) for _ in zip(*parts)])
    else:
        return np.mean(parts, axis=0)


def _gmm_entropy_integrand(y, mu, s2):
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
    return -p*log_p


class MCMC(Model):
    """
    Model which implements MCMC to produce a posterior over parameterized
    models.
    """
    def __init__(self, model, n=100, burn=100, skip=False, rng=None):
        self.__rng = rstate(rng)
        self.__ndata = 0
        self.__nburn = burn
        self.__nsamp = n
        self.__models = [model] if skip else self.__sample(model, True)

    def __iter__(self):
        return iter(self.__models)

    def get_models(self):
        return self.__models

    def get_hypers(self):
        hyper_dict = dict()
        for g in self.get_models():
            for comp_name, comp in g._objects:
                if comp_name not in hyper_dict.keys():
                    hyper_dict[comp_name] = dict()
                for hyper_name, hyper in comp._objects:
                    if hyper_name not in hyper_dict[comp_name].keys():
                        hyper_dict[comp_name][hyper_name] = []
                    hyper_dict[comp_name][hyper_name] += [hyper]

        for key1 in hyper_dict:
            for key2 in hyper_dict[key1]:
                hyper_dict[key1][key2] = np.array(hyper_dict[key1][key2])

        return hyper_dict

    def __sample(self, model, burn):
        """
        Return a list of models starting from `model`, potentially discarding
        samples if `burn` is True.
        """
        if burn and self.__nburn > 0:
            model = sample(model, self.__nburn, False, self.__rng)[-1]
        return sample(model, self.__nsamp, False, self.__rng)

    def add_data(self, X, Y):
        # add the data
        nprev = self.__ndata
        model = self.__models.pop()
        model.add_data(X, Y)
        self.__ndata += len(X)
        self.__models = self.__sample(model, self.__ndata > 2*nprev)

    def get_loglike(self):
        return np.mean([_.get_loglike() for _ in self.__models])

    def sample(self, X, size=None, latent=True, rng=None):
        rng = rstate(rng)
        model = self.__models[rng.randint(len(self.__models))]
        return model.sample(X, size, latent, rng)

    def predict(self, X, grad=False):
        # get predictions from each model
        predictions = np.array([_.predict(X) for _ in self.__models])

        if grad:
            raise NotImplementedError

        if predictions.shape == (len(X), 2, len(self.__models)):
            # if the output space is continuous then we will return a tuple
            # containing the mean and variance vectors
            mu = np.mean(predictions[:, 0], 0)
            s2 = np.mean(predictions[:, 1] + (predictions[:, 0] - mu)**2, 0)
            return mu, s2
        else:
            # otherwise a vector/array of class predictions
            return np.mean(predictions, axis=0)


    def get_tail(self, f, X, grad=False):
        parts = [m.get_tail(f, X, grad) for m in self.__models]
        return _integrate(parts, grad)

    def get_improvement(self, f, X, grad=False):
        parts = [m.get_improvement(f, X, grad) for m in self.__models]
        return _integrate(parts, grad)

    def get_entropy(self, X, grad=False):
        parts = [m.get_entropy(X, grad) for m in self.__models]
        return _integrate(parts, grad)

    def sample_f(self, n, rng=None):
        rng = rstate(rng)
        model = self.__models[rng.randint(len(self.__models))]
        return model.sample_f(n, rng)

    def get_ipes_entropy(self, X, grad=False):
        n = len(X)
        parts = np.array([_.predict(X) for _ in self.__models])
        mu = parts[:, 0, :]  # dimensions are models x data
        s2 = parts[:, 1, :]  # dimensions are models x data
        H = np.array([quad(_gmm_entropy_integrand,
                           np.min(mu[:, i]) - 100.0 * np.max(s2[:, i]),  # lower integration bound
                           np.max(mu[:, i]) + 100.0 * np.max(s2[:, i]),  # upper integration bound
                           args=(mu[:, i], s2[:, i]))[0] for i in range(n)])
        return H
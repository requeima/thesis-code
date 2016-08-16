"""
Perform parameter sampling.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..utils.misc import rstate

__all__ = ['sample']


def slice_sample(model, sigma=1.0, max_steps=1000, rng=None):
    """
    Implementation of a generic slice sampling step which takes a model
    instance and returns a new model instance.
    """
    def get_logp(theta):
        try:
            # make a copy of the model with the new parameters
            model_ = model.copy(theta)
        except ValueError:
            # if this happens then the parameters are outside the domain
            # defined either by the prior or the parameter itself, so their
            # probability should be zero.
            return -np.inf

        # compute the posterior probability; here we need to include the log-
        # determinant of the transform's jacobian
        return (model_.get_logprior() +
                model_.get_loglike() +
                model_.get_logjacobian())

    rng = rstate(rng)
    theta = model.hyper
    logp = get_logp(theta)

    for block in model.hyper_blocks:
        # sample a random direction
        direction = np.zeros_like(theta)
        direction[block] = rng.randn(len(block))
        direction /= np.sqrt(np.sum(direction**2))

        upper = sigma*rng.rand()
        lower = upper - sigma
        alpha = np.log(rng.rand())

        for _ in xrange(max_steps):
            if get_logp(theta + direction*lower) <= logp + alpha:
                break
            lower -= sigma

        for _ in xrange(max_steps):
            if get_logp(theta + direction*upper) <= logp + alpha:
                break
            upper += sigma

        while True:
            z = lower + (upper - lower) * rng.rand()
            theta_ = theta + direction * z
            logp_ = get_logp(theta_)
            if logp_ > logp + alpha:
                break
            elif z < 0:
                lower = z
            elif z > 0:
                upper = z
            else:
                raise RuntimeError("Slice sampler shrank to zero!")

        # make sure to update our starting point
        theta = theta_
        logp = logp_

    return model.copy(theta)


def sample(model, n, raw=False, rng=None):
    rng = rstate(rng)
    models = []
    for _ in xrange(n):
        model = slice_sample(model, rng=rng)
        models.append(model)
    if raw:
        models = np.array([m.hyper for m in models])
    return models

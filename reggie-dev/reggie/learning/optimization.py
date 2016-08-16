"""
Perform type-II maximum likelihood to fit the hyperparameters of a GP model.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.optimize as so

__all__ = ['optimize']


def optimize(model, raw=False):
    """
    Perform type-II maximum likelihood to fit GP hyperparameters.
    """
    # use a copy of the model so we don't modify it while we're optimizing
    model = model.copy()
    hyper = model.hyper

    if len(hyper) == 0:
        # this is kind of a stupid object to optimize, but we want to make sure
        # that we don't crash if someone decides to do this.
        hyper = np.array([])

    else:
        def objective(hyper):
            """
            Return the negative log marginal likelihood of the model and the
            gradient of this quantity wrt the parameters.
            """
            model.hyper = hyper
            logp0, dlogp0 = model.get_logprior(grad=True)
            logp1, dlogp1 = model.get_loglike(grad=True)
            return -(logp0 + logp1), -(dlogp0 + dlogp1)

        # optimize the model
        bounds = model.hyper_bounds
        hyper, _, _ = so.fmin_l_bfgs_b(objective, hyper, bounds=bounds)

    if raw:
        return hyper

    else:
        model.hyper = hyper
        return model

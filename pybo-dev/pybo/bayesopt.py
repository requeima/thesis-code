"""
Solver method for GP-based optimization which uses an inner-loop optimizer to
maximize some acquisition function, generally given as a simple function of the
posterior sufficient statistics.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import reggie as rg

from collections import namedtuple
from reggie.utils.misc import array2string

# each method/class defined exported by these modules will be exposed as a
# string to the solve_bayesopt method so that we can swap in/out different
# components for the "meta" solver.
from . import policies
from . import recommenders
from . import domains

from .utils import rstate

# exported symbols
__all__ = ['solve_bayesopt', 'init_model']


# create a namedtuple for returning info
Info = namedtuple('Info', 'x y r')


# MODEL INITIALIZATION ########################################################

def init_model(domain, X, Y, rng=None):
    """
    Initialize a model using an initial sample.
    """
    if not hasattr(domain, 'bounds'):
        raise ValueError('cannot construct a default model for the given '
                         'domain type')

    # define initial setting of hyper parameters
    sn2 = 1e-6
    rho = max(Y) - min(Y) if (len(Y) > 1) else 1.
    rho = 1. if (rho < 1e-1) else rho
    ell = 0.25 * np.array([b-a for (a, b) in domain.bounds])
    bias = np.mean(Y) if (len(Y) > 0) else 0.

    # initialize the base model
    model = rg.make_gp(sn2, rho, ell, bias)

    # define priors
    model.params['like']['sn2'].prior = rg.priors.Horseshoe(0.1)
    model.params['kern']['rho'].prior = rg.priors.LogNormal(np.log(rho), 1.)
    model.params['mean']['bias'].prior = rg.priors.Normal(bias, rho)

    for i, l in enumerate(ell):
        model.params['kern']['ell'][i].prior = rg.priors.Uniform(.01*l, 10*l)

    # initialize the MCMC inference meta-model and add data
    model = rg.MCMC(model, n=10, burn=100, skip=True, rng=rng)
    model.add_data(X, Y)

    return model


# THE QUERY/RECOMMEND STRATEGIES ##############################################

class IndexPolicy(object):
    """
    Class which implements a policy for Bayesian optimization as a wrapper
    around an acquisition function.
    """
    def __init__(self, domain, acquisition, kwp=None, kws=None):
        super(IndexPolicy, self).__init__()
        self._acquisition = getattr(policies, acquisition)
        self._domain = domain
        self._kwp = {} if (kwp is None) else kwp
        self._kws = {} if (kws is None) else kws
        self._index = None

    def __call__(self, model, X, rng=None):
        rng = rstate(rng)
        kwp = self._kwp
        kws = self._kws

        # construct the index and save it (for debugging purposes) and maximize
        # the index over whatever domain we have
        self._index = self._acquisition(model, self._domain, X, rng=rng, **kwp)
        xbest, _ = self._domain.solve(self._index, rng=rng, **kws)

        return xbest


class RecPolicy(object):
    """
    Policy which makes a recommendation.
    """
    def __init__(self, domain, recommender):
        super(RecPolicy, self).__init__()
        self._domain = domain
        self._recommender = getattr(recommenders, 'best_' + recommender)

    def __call__(self, model, X, Y, rng=None):
        return self._recommender(model, self._domain, X, Y, rng)


# THE BAYESOPT META SOLVER ####################################################

def solve_bayesopt(f,
                   domain,
                   model=None,
                   policy='EI',
                   recommender='observed',
                   niter=100,
                   verbose=False,
                   kwi=None,
                   kwp=None,
                   kws=None,
                   rng=None):
    """
    Maximize the given function using Bayesian Optimization.

    Args:
        f: function handle representing the objective function.
        bounds: bounds of the search space as a (d,2)-array.
        model: the Bayesian model instantiation.

        niter: horizon for optimization.
        init: the initialization component.
        policy: the acquisition component.
        solver: the inner-loop solver component.
        recommender: the recommendation component.
        rng: either an RandomState object or an integer used to seed the state;
             this will be fed to each component that requests randomness.
        callback: a function to call on each iteration for visualization.

    Note that the modular way in which this function has been written allows
    one to also pass parameters directly to some of the components. This works
    for the `init`, `policy`, `solver`, and `recommender` inputs. These
    components can be passed as either a string, a function, or a 2-tuple where
    the first item is a string/function and the second is a dictionary of
    additional arguments to pass to the component.

    Returns:
        A numpy record array containing a trace of the optimization process.
        The fields of this array are `x`, `y`, and `xbest` corresponding to the
        query locations, outputs, and recommendations at each iteration. If
        ground-truth is known an additional field `fbest` will be included.
    """
    # interpret the domain as a list of bounds if necessary
    if not isinstance(domain, domains.domains.Domain):
        domain = domains.Box(domain)

    rng = rstate(rng)
    kwi = {} if (kwi is None) else kwi

    # sample any initial points.
    X = list(domain.init(rng=rng, **kwi))   # FIXME: ignores init string
    Y = [f(x) for x in X]                   # FIXME: should save data
    R = []

    # either create the default model or add the initial data.
    if model is None:
        model = init_model(domain, X, Y)
    else:
        model.add_data(X, Y)

    # initialize the policies
    policy = IndexPolicy(domain, policy, kwp, kws)
    recommender = RecPolicy(domain, recommender)

    for i in xrange(niter):
        # get the next query point, make an observation, record it, and find
        # the recommendation.
        x = policy(model, X, rng)
        y = f(x)
        model.add_data(x, y)
        r = recommender(model, X, Y, rng)

        # save progress
        X.append(x)
        Y.append(y)
        R.append(r)

        # print out the progress if requested.
        if verbose:
            print('i={:03d}, x={:s}, y={:s}, xbest={:s}'
                  .format(i,
                          array2string(x, sign=True),
                          array2string(y, sign=True),
                          array2string(r, sign=True)))

    # map everything to arrays.
    info = Info(*map(np.array, (X, Y, R)))

    return R[-1], model, info

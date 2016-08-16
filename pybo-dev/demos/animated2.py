"""
Animated demo showing progress of Bayesian optimization on a simple
two-dimensional function.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import reggie as rg
import ezplot as ez

from pybo.bayesopt import IndexPolicy, RecPolicy
from pybo.domains import Grid

__all__ = []


def f(x):
    """
    Test function we'll optimize. This is the much-used (and perhaps overused)
    2d Branin function.
    """
    x = np.array(x, ndmin=2)
    y = (x[:, 1]-(5.1/(4*np.pi**2))*x[:, 0]**2+5*x[:, 0]/np.pi-6)**2
    y += 10*(1-1/(8*np.pi))*np.cos(x[:, 0])+10
    # NOTE: this rescales branin by 10 to make it more manageable.
    y /= 10.
    return -np.squeeze(y)


# define the bounds over which we'll optimize, the optimal x for comparison,
# and a sequence of test points
bounds = [[-5, 10.], [0, 15]]
xopt = np.array([np.pi, 2.275])

# construct the domain and policies
domain = Grid(bounds, 100)
policy = IndexPolicy(domain, 'EI')
recommender = RecPolicy(domain, 'observed')

# get initial data and some test points.
X = list(domain.init())
Y = [f(x_) for x_ in X]
F = []

# initialize the model
model = rg.make_gp(0.01, 10, [1., 1.], 0)
model.params['like']['sn2'].prior = rg.priors.Uniform(0.005, 0.015)
model.params['kern']['rho'].prior = rg.priors.LogNormal(0, 3)
model.params['kern']['ell'].prior = rg.priors.LogNormal(0, 3)
model.params['mean']['bias'].prior = rg.priors.Normal(0, 20)

# make the meta-model and add data
model = rg.MCMC(model, n=20, skip=True)
model.add_data(X, Y)

# create a new figure
fig = ez.figure(figsize=(10, 6))
x1, x2 = np.meshgrid(np.linspace(*bounds[0], num=100),
                     np.linspace(*bounds[1], num=100))

while True:
    # get the recommendation and the next query
    xbest = recommender(model, X, Y)
    xnext = policy(model, X)
    ynext = f(xnext)

    # evaluate the posterior before updating the model for plotting
    mu, _ = model.predict(domain.X)

    # record our data and update the model
    X.append(xnext)
    Y.append(ynext)
    F.append(f(xbest))
    model.add_data(xnext, ynext)

    fig.clear()
    ax1 = fig.add_subplotspec((2, 2), (0, 0), hidex=True)
    ax2 = fig.add_subplotspec((2, 2), (1, 0), hidey=True, sharex=ax1)
    ax3 = fig.add_subplotspec((2, 2), (0, 1), rowspan=2)

    # plot the posterior and data
    ax1.contourf(x1, x2, mu.reshape(x1.shape), alpha=0.4)
    X_ = np.array(X)
    ax1.scatter(X_[:-1, 0], X_[:-1, 1], marker='.')
    ax1.scatter(xbest[0], xbest[1], linewidths=3, marker='o', color='r')
    ax1.scatter(xnext[0], xnext[1], linewidths=3, marker='o', color='g')
    ax1.set_xlim(*bounds[0])
    ax1.set_ylim(*bounds[1])
    ax1.set_title('current model (xbest and xnext)')

    # plot the acquisition function
    ax2.contourf(x1, x2, policy._index(domain.X).reshape(x1.shape), alpha=0.5)
    ax2.scatter(xbest[0], xbest[1], linewidths=3, marker='o', color='r')
    ax2.scatter(xnext[0], xnext[1], linewidths=3, marker='o', color='g')
    ax2.set_xlim(*bounds[0])
    ax2.set_ylim(*bounds[1])
    ax2.set_title('current policy (xnext)')

    # plot the latent function at recomended points
    ax3.axhline(f(xopt))
    ax3.plot(F)
    ax3.set_ylim(-1., 0.)
    ax3.set_title('value of recommendation')

    # draw
    fig.canvas.draw()
    ez.show(block=False)

"""
Example showing the results of conditioning a GP on the location, but not the
value, of a maximizer.
"""

import numpy as np

from ezplot import figure, show
from reggie import make_gp


if __name__ == '__main__':
    # seed the rng
    rng = np.random.RandomState(0)
    xstar = 0.2
    fstar = 1.8

    # generate data from a GP prior.
    gp = make_gp(sn2=0.1, rho=1, ell=0.05, kernel='matern3')
    X = rng.rand(20, 1)
    Y = gp.sample(X, latent=False, rng=rng)

    # create a new GP, add data, and optimize; note the different kernel
    gp = make_gp(sn2=0.1, rho=1, ell=0.25)
    gp.add_data(X, Y)
    gp.optimize()

    # get the test locations.
    z = np.linspace(X.min(), X.max(), 200)

    # get the "prior" and "posterior" predictions.
    mu, s2 = gp.predict(z[:, None])
    xmu, xs2 = gp.condition_xstar(xstar).predict(z[:, None])
    fmu, fs2 = gp.condition_fstar(fstar).predict(z[:, None])

    # create a new figure
    fig = figure()
    ax1 = fig.add_subplot(121, hidexy=True)
    ax2 = fig.add_subplot(122, hidexy=True, sharey=ax1)

    ax1.plot_banded(z, mu, 2*np.sqrt(s2), label='before')
    ax1.plot_banded(z, xmu, 2*np.sqrt(xs2), label='after')
    ax1.scatter(X.ravel(), Y)
    ax1.axvline(xstar)
    ax1.set_title('conditioning on $\mathbf{x}_\star$')

    ax2.plot_banded(z, mu, 2*np.sqrt(s2), label='before')
    ax2.plot_banded(z, fmu, 2*np.sqrt(fs2), label='after')
    ax2.scatter(X.ravel(), Y)
    ax2.axhline(fstar)
    ax2.legend(loc=4)
    ax2.set_title('conditioning on $f_\star$')

    # draw it
    ax1.figure.canvas.draw()
    ax2.figure.canvas.draw()
    show()

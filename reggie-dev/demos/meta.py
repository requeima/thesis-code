"""
Demo showing a meta-model using a GP and MCMC over the hyperparameters.
"""

import numpy as np
import ezplot as ez
import reggie as rg

# create a gp to sample data from
gp = rg.make_gp(0.1, 1.0, 0.1, kernel='matern1')

# generate random data
rng = np.random.RandomState(0)
X = rng.uniform(-2, 2, size=(20, 1))
Y = gp.sample(X, latent=False, rng=rng)

# create a new GP and and set priors over the hyperparameters
model = rg.make_gp(1, 1, 1)
model.params['like'].prior = rg.core.priors.Uniform(1e-4, 10)
model.params['kern'].prior = rg.core.priors.Uniform(1e-4, 10)
model.params['mean'].prior = rg.core.priors.Uniform(-10, 10)
model = rg.MCMC(model, skip=True, rng=rng)

# add the data and set up the meta-model
model.add_data(X, Y)

# get the posterior moments
x = np.linspace(X.min(), X.max(), 500)
mu, s2 = model.predict(x[:, None])

# plot the posterior
ax = ez.figure().gca()
ax.plot_banded(x, mu, 2*np.sqrt(s2), label='posterior mean')
ax.scatter(X.ravel(), Y, label='observed data')
ax.legend(loc=0)
ax.set_title('Basic GP with sampled hyperparameters')
ax.set_xlabel('inputs, X')
ax.set_ylabel('outputs, Y')

# draw and show it
ax.figure.canvas.draw()
ez.show()

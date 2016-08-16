"""
Interface for parameterized objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import itertools as it
import copy
import tabulate
import collections
import warnings

from ..utils.misc import array2string, setter
from .domains import Domain
from .priors import Prior

__all__ = ['Parameterized']


#==============================================================================
# Parameterized
#==============================================================================

class Parameterized(object):
    """
    Base class for objects which can be parameterized.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the object given a variable number of parameters and any
        additional kwargs. The parameters should be passed as tuples that are
        of the form::

            (name, parameterized, [cls])
            (name, array, domain, [shape])

        where the square-brackets indicate optional arguments. In both cases
        `name` must be a unique string. The `parameterized` argument must be an
        instance of some `Parameterized` object and if given `cls` must be a
        subclass of `Parameterized` so that an error can be raised if
        `parameterized` is not a valid instance of `cls`.

        The `array` parameter must be an instance of `numpy.ndarray`, `domain`
        must be a `Domain` instance (e.g. `Real` or `Positive`), and `shape`
        must be a valid `numpy` shape.

        Any passed kwargs will be stored for use by the `repr` function, but
        otherwise have no effect.
        """
        # check that the args are valid as defined in the docstring above. The
        # resulting list of args should be (name, param, domain, size) where
        # domain is None if the parameter is a Parameterized object.
        args = list(check_args(args))
        size = sum(size for (_, _, _, size) in args)

        self.__params = np.empty(size)
        self.__domains = np.empty(size, dtype=object)
        self.__priors = np.empty(size, dtype=object)
        self.__transforms = np.empty(size, dtype=object)
        self.__blocks = np.zeros(size, dtype=int)
        self.__shape = ()
        self.__objects = []
        self.__kwargs = kwargs

        b = 0
        for name, param, domain, size in args:
            a = b
            b = a + size

            # if the domain is None then this parameter is a collection of
            # parameters given by another Parameterized object
            if domain is None:
                # pylint: disable=W0212
                self.__params[a:b] = param.__params
                self.__domains[a:b] = param.__domains
                self.__priors[a:b] = param.__priors
                self.__transforms[a:b] = param.__transforms
                self.__blocks[a:b] = param.__blocks
                self.__shape += ((name, param.__shape),)

                # this will make a deepcopy of the object, but make sure to
                # "delete" access to the parameter arrays and also make the
                # object arrays point towards the new memory storage.
                memo = dict()
                memo[id(param.__params)] = None
                memo[id(param.__domains)] = None
                memo[id(param.__priors)] = None
                memo[id(param.__transforms)] = None
                memo[id(param.__blocks)] = None
                memo[id(param.__shape)] = None
                memo.update(param.__get_memo(self.__params[a:b]))
                self.__objects.append((name, copy.deepcopy(param, memo)))

            else:
                self.__params[a:b] = param.flat
                self.__domains[a:b] = domain
                self.__priors[a:b] = None
                self.__transforms[a:b] = domain.transform
                self.__blocks[a:b] = 0
                self.__shape += ((name, param.shape),)
                self.__objects.append(
                    (name, self.__params[a:b].reshape(param.shape)))

        # save local access to each parameter object
        for (name, param) in self.__objects:
            setattr(self, '_' + name, param)

        # filter out any empty shapes (those that are None) and if the
        # resulting shape is empty replace it by None.
        self.__shape = tuple(s for s in self.__shape if s[1] is not None)
        self.__shape = None if (self.__shape == ()) else self.__shape
        self._objects = self.__objects

    def __repr__(self):
        typename = self.__class__.__name__
        parts = []
        for name, obj in self.__objects:
            if isinstance(obj, Parameterized):
                parts.append('{}={!r}'.format(name, obj))
            else:
                parts.append('{}={}'.format(name, array2string(obj)))
        for name, value in self.__kwargs.items():
            parts.append('{}={!r}'.format(name, value))
        return typename + '(' + ', '.join(parts) + ')'

    def __deepcopy__(self, memo):
        params = copy.deepcopy(self.__params, memo)
        if params is not None:
            memo.update(self.__get_memo(params))
        obj = type(self).__new__(type(self))
        for key, val in self.__dict__.items():
            setattr(obj, key, copy.deepcopy(val, memo))
        return obj

    def __get_memo(self, params):
        # pylint: disable=C0111
        # Recursively find all of the arrays defined inside self and the
        # objects inside self.__objects and return a dictionary which maps the
        # ids of these arrays to new views into `params`. This is designed to
        # be used by `__deepcopy__` as part of the `memo` parameter in order to
        # make sure that copying makes views into `self.__params`.
        def get_arrays(self):
            # pylint: disable=C0111
            # Inner function which recursively finds the arrays in an object.
            for _, obj in self.__objects:
                if isinstance(obj, Parameterized):
                    for _ in get_arrays(obj):
                        yield _
                else:
                    yield obj
        memo = dict()
        a = 0
        for array in get_arrays(self):
            b = a + array.size
            memo[id(array)] = params[a:b].reshape(array.shape)
            a = b
        return memo

    def copy(self, hyper=None):
        """
        Return a copy of the object.
        """
        other = copy.deepcopy(self)
        if hyper is not None:
            other.hyper = hyper
        return other

    @property
    def params(self):
        """
        Proxy object that allows for the setting of parameter values, priors,
        etc. For example the following code::

            obj.params.value = 1
            obj.params.prior = Uniform(0, 1)
            obj.params['foo']['bar'].value = 12

        will set values and priors for the given object. One can also obtain
        a human-readable description of the parameters using
        `obj.params.describe()`.
        """
        return ParameterProxy(self.__params, self.__domains, self.__priors,
                              self.__transforms, self.__blocks, self.__shape,
                              self._update)

    @property
    def _nhyper(self):
        """
        Internal property which returns the number of hyperparameters. Note
        that this size is computed BEFORE any wrapping, fixing of parameters,
        etc. So the self._nhyper may not be equal to len(self.hyper).
        """
        return self.__params.size

    @property
    def hyper(self):
        """
        The hyperparameter vector. Note that this property returns a copy of
        the internal memory so that things like `param.hyper[0] = 1` won't
        actually make changes. This is actually a good thing because allowing
        that would bypass any of the transformations (which we don't want).
        """
        if self.__params is None:
            raise AttributeError('hyperparameters of a sub-model cannot be '
                                 'accessed')
        iterable = (t.get_transform(x) for (x, t) in
                    it.izip(self.__params, self.__transforms))
        return np.fromiter(iterable, float)

    @hyper.setter
    def hyper(self, value):
        # pylint: disable=C0111
        if self.__params is None:
            raise AttributeError('hyperparameters of a sub-model cannot be '
                                 'accessed')

        # check that the assigned values lie in the domain
        z = it.izip(value, self.__domains, self.__priors, self.__transforms)
        i = (v in t.get_image(d if p is None else p.domain)
             for (v, d, p, t) in z)
        if not all(i):
            raise ValueError('hyperparameter assignment is invalid')

        # set the value which may need to be inverse-transformed
        iterable = (t.get_inverse(v) for (v, t) in
                    it.izip(value, self.__transforms))
        self.__params[:] = np.fromiter(iterable, float)

        # do any post-processing
        self._update()

    @property
    def hyper_bounds(self):
        """
        The hyperparameter bounds as a list of 2-tuples. These are defined by a
        combination of each parameters' domain, the domain of each prior, and
        the image of this domain under the transformation (if any).
        """
        if self.__params is None:
            raise AttributeError('hyperparameters of a sub-model cannot be '
                                 'accessed')

        # get the image of the domain and return its bounds
        return [t.get_image(d if (p is None) else p.domain).bounds
                for (d, p, t) in zip(self.__domains,
                                     self.__priors,
                                     self.__transforms)]

    @property
    def hyper_blocks(self):
        """
        Return a list whose ith element contains indices for the parameters
        which make up the ith block.
        """
        blocks = dict()
        for i, block in enumerate(self.__blocks):
            blocks.setdefault(block, []).append(i)
        return blocks.values()

    def _update(self):
        """
        Method which should be called everytime that the parameters change.
        This should be overridden in any classes that are children of
        Parameterized.
        """
        pass

    def _wrap_gradient(self, gradient):
        """
        Internal method to wrap a gradient which multiplies the gradient by a
        factor (via the chain rule) which arises due to any transformations.
        If the object is included as a sub-model of any other Parameterized
        object then its `self.__params` and `self.__transforms` arrays should
        be `None` and hence this function should do nothing.
        """
        # return immediately if we don't store the parameters locally.
        if self.__params is None:
            return gradient

        # make sure the gradient is an ndarray and get the sequence of
        # gradfactors
        gradient = np.array(gradient, copy=False)
        iterable = (t.get_gradfactor(x)
                    for (x, t) in it.izip(self.__params, self.__transforms))

        # the transposes make sure that broadcasting is ok.
        return (gradient.T * np.fromiter(iterable, float)).T

    def get_logprior(self, grad=False):
        """
        Return the log probability of hyperparameter assignments under the
        prior. If requested, also return the gradient of this probability with
        respect to the parameter values.
        """
        if self.__params is None:
            raise AttributeError('hyperparameters of a sub-model cannot be '
                                 'accessed')
        if grad:
            zipped = it.izip(self.__params, self.__priors, self.__transforms)
            logprior = 0.0
            dlogprior = np.empty(self._nhyper)
            for i, (d, p, t) in enumerate(zipped):
                f, g = (0.0, 0.0) if (p is None) else p.get_logprior(d, True)
                logprior += f
                dlogprior[i] = g * t.get_gradfactor(d)
            return logprior, dlogprior
        else:
            zipped = it.izip(self.__params, self.__priors)
            logprior = np.sum(p.get_logprior(d)
                              for (d, p) in zipped if p is not None)
            return logprior

    def get_logjacobian(self):
        """
        Return the log-Jacobian due to any transformations of the space. This
        corresponds to a sum of the log-gradfactor associated with
        transformations of any parameter.
        """
        if self.__params is None:
            raise AttributeError('hyperparameters of a sub-model cannot be '
                                 'accessed')

        # get the log sum of each transform's gradfactor
        return sum(np.log(t.get_gradfactor(d))
                   for (d, t) in it.izip(self.__params, self.__transforms))


#==============================================================================
# check_args. this is just a helper function which simplifies the init method
# from Parameterized.
#==============================================================================

def check_args(args):
    """
    Parse the list of parameters passed to a Parameterized object. This will
    yield a list of (name, param, domain, size) tuples. The input `args` should
    be a list of tuples of the form::

        (name, parameterized, [cls])
        (name, array, domain, [shape])

    and an error should be raised if this format is not followed.
    """
    def check_arg(arg):
        # pylint: disable=C0111
        # check that the argument matches the format detailed in the docstring
        # and if so return a (name, param, domain, shape) tuple. In the
        # returned tuple, domain will either be a Parameterized subclass or a
        # valid Domain.
        assert isinstance(arg, collections.Sequence) and len(arg) in (2, 3, 4)
        name = arg[0]
        param = arg[1]
        domain = Parameterized
        shape = None
        assert isinstance(name, str)
        if len(arg) == 3:
            domain = arg[2]
            assert (isinstance(domain, Domain) or
                    isinstance(domain, type) and
                    issubclass(domain, Parameterized))
        elif len(arg) == 4:
            domain = arg[2]
            shape = arg[3]
            assert (isinstance(domain, Domain) and
                    isinstance(shape, collections.Sequence) and
                    all(isinstance(_, (int, str)) for _ in shape))
        return name, param, domain, shape

    # initialize a dictionary that we'll incrementally build up in order to
    # validate named dimensions of our arrays.
    dims = dict()

    for arg in args:
        # these checks may raise AssertionError, but these will fail only if
        # the arguments passed to the init method are malformed. the user of a
        # Parameterized class should never see these exceptions.
        name, param, domain, shape = check_arg(arg)

        if isinstance(domain, Domain):
            shape = () if (shape is None) else shape
            shape = (shape,) if isinstance(shape, (int, str)) else shape
            ndmin = len(shape)

            # attempt to force the parameter into an array.
            try:
                param = np.array(param, dtype=float, copy=False, ndmin=ndmin)
            except (TypeError, ValueError):
                msg = "Parameter '{}' is not an array (or array-like)"
                raise ValueError(msg.format(name))

            # construct the real shape. this will infer the value of any named
            # dimension identifiers if they have been previously used.
            shape_ = tuple(
                (dims.setdefault(d, d_) if isinstance(d, str) else d)
                for (d, d_) in zip(shape, param.shape))

            # check that the shape matches
            if param.shape != shape_:
                msg = "Parameter '{}' should have shape ({})"
                shape = ', '.join(str(_) for _ in shape)
                raise ValueError(msg.format(name, shape))

            # check that the values are in the given domain
            if not all(v in domain for v in param.flat):
                msg = "Parameter '{}' is not in the domain '{}'"
                raise ValueError(msg.format(name, domain))

            yield name, param, domain, param.size
        else:
            if not isinstance(param, domain):
                msg = "Parameter '{}' must be an object of type '{}'"
                raise ValueError(msg.format(name, domain.__name__))
            # pylint: disable=W0212
            yield name, param, None, param._nhyper


#==============================================================================
# ParameterProxy. this is just a helper object which should be return by
# Parameterized.params and which will allow the priors/values/etc. of a
# Parameterized object to be changed without exposing the internal arrays and
# adding some additional checks.
#==============================================================================

class ParameterProxy(object):
    """
    Proxy object which allows modification of parameter values, priors, etc.
    """
    def __init__(self, params, domains, priors, transforms, blocks, shape,
                 callback):
        self.__params = params
        self.__domains = domains
        self.__priors = priors
        self.__transforms = transforms
        self.__blocks = blocks
        self.__shape = shape
        self.__callback = callback

    def describe(self):
        """
        Print a description of the parameters.
        """
        def get_names(shape, namespace=()):
            # pylint: disable=C0111
            if shape == ():
                yield ':'.join(namespace)
            elif all(isinstance(_, int) for _ in shape):
                name = ':'.join(namespace)
                for ijk in np.ndindex(shape):
                    yield name + '[' + ','.join(map(str, ijk)) + ']'
            else:
                for name, subshape in shape:
                    for _ in get_names(subshape, namespace + (name,)):
                        yield _

        head = ['name', 'domain', 'value', 'block', 'prior']
        data = zip(get_names(self.__shape),
                   self.__domains,
                   self.__params,
                   self.__blocks,
                   self.__priors)

        print(tabulate.tabulate(data, head))

    def __getitem__(self, key):
        def shape2size(shape):
            # pylint: disable=C0111
            # get the size of a hierarchically defined shape tuple.
            if shape == ():
                return 1
            elif all(isinstance(_, int) for _ in shape):
                return np.prod(shape)
            else:
                return sum(shape2size(subshape) for (_, subshape) in shape)

        if self.__shape == ():
            raise IndexError('scalar parameters cannot be indexed')

        elif all(isinstance(_, int) for _ in self.__shape):
            key = (key,) if not isinstance(key, tuple) else key
            if not all(isinstance(_, int) for _ in key):
                raise IndexError('indices must be integers')
            if len(key) != len(self.__shape):
                raise IndexError('the parameter requires {:d} indices; {:d} '
                                 'were given'.format(len(self.__shape),
                                                     len(key)))
            a = np.ravel_multi_index(key, self.__shape)
            b = a + 1
            shape = ()

        else:
            b = 0
            for name, shape in self.__shape:
                a = b
                b = a + shape2size(shape)
                if name == key:
                    break
            else:
                raise KeyError("unknown key '{}'".format(key))

        return ParameterProxy(self.__params[a:b], self.__domains[a:b],
                              self.__priors[a:b], self.__transforms[a:b],
                              self.__blocks[a:b], shape, self.__callback)

    @property
    def value(self):
        """
        Value of the given set of parameters.
        """
        return self.__params.copy()

    @value.setter
    def value(self, val):
        # pylint: disable=C0111
        val = np.broadcast_to(val, self.__params.size)
        dom = [d if (p is None) else p.domain
               for (p, d) in zip(self.__priors, self.__domains)]
        if not all(v in d for (v, d) in zip(val, dom)):
            raise ValueError('value is not in the domain of the given '
                             'parameters')
        self.__params[:] = val
        self.__callback()

    @setter
    def prior(self, prior):
        """
        Write-only property used to set the prior of the given parameters.
        """
        if prior is not None and not isinstance(prior, Prior):
            raise ValueError('prior must be `None` or a `Prior` instance')
        if prior is not None and not all(prior.domain <= d
                                         for d in self.__domains):
            raise ValueError('priors are not valid for the given '
                             'parameters')
        self.__priors[:] = prior

        if not all(v in prior.domain for v in self.__params):
            message = 'parameters lie outside prior support; clipping values'
            warnings.warn(message, stacklevel=2)
            self.__params[:] = [prior.domain.project(v) for v in self.__params]
            self.__callback()

    @setter
    def block(self, block):
        """
        Write-only property used to set the block of the given parameters.
        """
        if not isinstance(block, int):
            raise ValueError('the block must be an integer')
        self.__blocks[:] = block

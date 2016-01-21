from __future__ import absolute_import
from . import backend as K

import numpy as np

class Constraint(object):
    def __call__(self, p):
        return p

    def get_config(self):
        return {"name": self.__class__.__name__}


class MaxNorm(Constraint):
    def __init__(self, m=2):
        self.m = m

    def __call__(self, p):
        norms = K.sqrt(K.sum(K.square(p), axis=0))
        desired = K.clip(norms, 0, self.m)
        p = p * (desired / (1e-7 + norms))
        return p

    def get_config(self):
        return {"name": self.__class__.__name__,
                "m": self.m}


class MaxFrobeniusNorm(Constraint):
    def __init__(self, m=1):
        self.m = m

    def __call__(self, p):
        norm = K.sqrt(K.sum(K.square(p)))
        desired = K.clip(norm, 0, self.m)
        p = p * (desired / (1e-7 + norm))
        return p

    def get_config(self):
        return {"name": self.__class__.__name__,
                "m": self.m}


class NonNeg(Constraint):
    def __call__(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p


class UnitNorm(Constraint):
    def __call__(self, p):
        return p / K.sqrt(K.sum(K.square(p), axis=-1, keepdims=True))


def _simplex_projection(v, s=1):
    assert s > 0
    # check that v is a vector (theano specific code)
    assert K.ndim(v) == 1

    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = K.sort(v)[::-1]
    cssv = K.cumsum(u)
    rho = K.nonzero(K.ge(u * K.arange(1, K.shape(v)[0]+1), cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    p = K.maximum(v - theta, 0)

    #if v.sum() == s and np.alltrue(v >= 0):
    #    # best projection: itself!
    #    return v
    # else we will return the projection p
    #return K.ifelse(K.and_(K.equal(v.sum(), s), K.all(K.ge(v, 0))),
    #                v, p)
    return K.cast(p, 'float32')


def _filter_simplex_projection(W, s=1):
    assert s > 0

    n, c, h, w = W.shape
    # check that c == 1

    u = K.sort(W, axis=2)[:, :, ::-1, :]
    cssv = K.cumsum(u, axis=2)

    cond = (u * K.arange(1, h+1)[:, np.newaxis]) > (cssv - s)
    # This finds the largest index that is still True by doing an argmax on the
    # flipped array, as argmax returns the earliest index that is the max.
    rho = (h - K.argmax(cond[:, :, ::-1, :], axis=2) - 1).flatten()
    indices = (K.repeat_elements(K.arange(n), w, axis=0),
               0,
               rho,
               K.tile(K.arange(w), n))
    theta = (((cssv[indices] - s) / (rho + 1.0))
                .reshape((n, 1, 1, w)))
    p = K.maximum(W - theta, 0.0)
    return K.cast(p, 'float32')


class PWMSimplex(Constraint):
    def __call__(self, p):
        return _filter_simplex_projection(p)


class Simplex(Constraint):
    def __init__(self, m=1):
        self.m = m

    def __call__(self, p):
        original_shape = K.shape(p)
        projection = _simplex_projection(K.flatten(p), self.m)
        return K.reshape(projection, original_shape)


identity = Constraint
maxnorm = MaxNorm
maxfrobeniusnorm = MaxFrobeniusNorm
nonneg = NonNeg
unitnorm = UnitNorm
pwmsimplex = PWMSimplex
simplex = Simplex

from .utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'constraint', instantiate=True, kwargs=kwargs)

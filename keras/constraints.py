from __future__ import absolute_import
from . import backend as K

import numpy as np

class Constraint(object):
    def __call__(self, p):
        return p

    def get_config(self):
        return {'name': self.__class__.__name__}


class MaxNorm(Constraint):
    '''Constrain the weights incident to each hidden unit to have a norm less than or equal to a desired value.

    # Arguments
        m: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms. For instance,
            in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
            set `axis` to `0` to constrain each weight vector of length (input_dim).
            In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
            set `axis` to `1` to constrain each weight vector of length (input_dim),
            i.e. constrain the filters incident to the `max` operation.
            In a `Convolution2D` layer with the Theano backend, the weight tensor
            has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
            to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
            In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
            has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
            to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, m=2, axis=0):
        self.m = m
        self.axis = axis

    def __call__(self, p):
        norms = K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True))
        desired = K.clip(norms, 0, self.m)
        p = p * (desired / (K.epsilon() + norms))
        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m,
                'axis': self.axis}


class NonNeg(Constraint):
    '''Constrain the weights to be non-negative.
    '''
    def __call__(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p


class UnitNorm(Constraint):
    '''Constrain the weights incident to each hidden unit to have unit norm.

    # Arguments
        axis: integer, axis along which to calculate weight norms. For instance,
            in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
            set `axis` to `0` to constrain each weight vector of length (input_dim).
            In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
            set `axis` to `1` to constrain each weight vector of length (input_dim),
            i.e. constrain the filters incident to the `max` operation.
            In a `Convolution2D` layer with the Theano backend, the weight tensor
            has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
            to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
            In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
            has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
            to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).
    '''
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        return p / (K.epsilon() + K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}


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
    return K.ifelse(K.and_(K.equal(v.sum(), s), K.all(K.ge(v, 0))),
                    v, p)


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

identity = Constraint
maxnorm = MaxNorm
nonneg = NonNeg
unitnorm = UnitNorm
pwmsimplex = PWMSimplex

from .utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'constraint',
                           instantiate=True, kwargs=kwargs)

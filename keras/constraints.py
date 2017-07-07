from __future__ import absolute_import
from . import backend as K
from .utils.generic_utils import get_from_module


class Constraint(object):

    def __call__(self, p):
        return p

    def get_config(self):
        return {'name': self.__class__.__name__}


class CurvatureConstraint(Constraint):
    """ Specific to SeparableFC
        Constrains the second differences of weights in W_pos.

    # Arguments
        m: the maximum allowed curvature which constrains
            second differences of adjacent weights in the length dimension
            to be within the specified range
    """

    def __init__(self, m=1.0):
        self.m = float(m)

    def __call__(self, p):
        import numpy as np
        mean_p = K.mean(p, axis=1)
        (num_output, length) = K.int_shape(p)
        diff1 = p[:, 1:] - p[:, :-1]
        mean_diff1 = K.mean(diff1, axis=1)
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        desired_diff2 = K.clip(diff2, -1.0 * self.m, self.m)

        il1 = np.triu_indices(length - 2)
        mask1 = np.ones((num_output, length - 1, length - 2))
        mask1[:, il1[0], il1[1]] = 0.0
        kmask1 = K.variable(value=mask1)
        mat1 = kmask1 * K.repeat_elements(K.expand_dims(desired_diff2, 1), length - 1, 1)
        desired_diff1 = K.squeeze(K.squeeze(
            K.dot(mat1, K.ones((1, length - 2, num_output)))[:, :, :1, :1], axis=2), axis=2)
        desired_diff1 += K.repeat_elements(K.expand_dims(
            mean_diff1 - K.mean(desired_diff1, axis=1), -1), length - 1, axis=1)

        il2 = np.triu_indices(length - 1)
        mask2 = np.ones((num_output, length, length - 1))
        mask2[:, il2[0], il2[1]] = 0.0
        kmask2 = K.variable(value=mask2)
        mat2 = kmask2 * K.repeat_elements(K.expand_dims(desired_diff1, 1), length, 1)
        desired_p = K.squeeze(K.squeeze(
            K.dot(mat2, K.ones((1, length - 1, num_output)))[:, :, :1, :1], axis=2), axis=2)
        desired_p += K.repeat_elements(K.expand_dims(
            mean_p - K.mean(desired_p, axis=1), -1), length, axis=1)

        return desired_p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m}  
    
  
class MaxNorm(Constraint):
    """MaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.

    # Arguments
        m: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Convolution2D` layer with `dim_ordering="tf"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, m=2, axis=0):
        self.m = m
        self.axis = axis

    def __call__(self, p):
        norms = K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True))
        desired = K.clip(norms, 0, self.m)
        p *= (desired / (K.epsilon() + norms))
        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m,
                'axis': self.axis}


class NonNeg(Constraint):
    """Constrains the weights to be non-negative.
    """

    def __call__(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p


class UnitNorm(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.

    # Arguments
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Convolution2D` layer with `dim_ordering="tf"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        return p / (K.epsilon() + K.sqrt(K.sum(K.square(p),
                                               axis=self.axis,
                                               keepdims=True)))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}


class MeanNormConv1D(Constraint):
    '''Constrain the weights to have mean norm.

    # Arguments
        axis: integer, axis along which to calculate mean norms. 
    '''
    def __call__(self, p):
        return p - K.mean(p, axis=-2)[:,:,None,:]

    def get_config(self):
        return {'name': self.__class__.__name__}

class MinMaxNorm(Constraint):
    """MinMaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have the norm between a lower bound and an upper bound.

    # Arguments
        low: the minimum norm for the incoming weights.
        high: the maximum norm for the incoming weights.
        rate: rate for enforcing the constraint: weights will be
            rescaled to yield (1 - rate) * norm + rate * norm.clip(low, high).
            Effectively, this means that rate=1.0 stands for strict
            enforcement of the constraint, while rate<1.0 means that
            weights will be rescaled at each step to slowly move
            towards a value inside the desired interval.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Convolution2D` layer with `dim_ordering="tf"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """
    def __init__(self, low=0.0, high=1.0, rate=1.0, axis=0):
        self.low = low
        self.high = high
        self.rate = rate
        self.axis = axis

    def __call__(self, p):
        norms = K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True))
        desired = self.rate * K.clip(norms, self.low, self.high) + (1 - self.rate) * norms
        p *= (desired / (K.epsilon() + norms))
        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'low': self.low,
                'high': self.high,
                'rate': self.rate,
                'axis': self.axis}


# Aliases.
maxnorm = MaxNorm
nonneg = NonNeg
unitnorm = UnitNorm


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'constraint',
                           instantiate=True, kwargs=kwargs)

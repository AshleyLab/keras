# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..layers.core import Layer
import numpy as np
import pdb 

def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride


class Convolution1D(Layer):
    '''Convolution operator for filtering neighborhoods of one-dimensional inputs.
    When using this layer as the first layer in a model,
    either provide the keyword argument `input_dim`
    (int, e.g. 128 for sequences of 128-dimensional vectors),
    or `input_shape` (tuple of integers, e.g. (10, 128) for sequences
    of 10 vectors of 128-dimensional vectors).

    # Input shape
        3D tensor with shape: `(samples, steps, input_dim)`.

    # Output shape
        3D tensor with shape: `(samples, new_steps, nb_filter)`.
        `steps` value might have changed due to padding.

    # Arguments
        nb_filter: Number of convolution kernels to use
            (dimensionality of the output).
        filter_length: The extension (spatial or temporal) of each filter.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample_length: factor by which to subsample output.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
    '''
    input_ndim = 3

    def __init__(self, nb_filter, filter_length,
                 init='uniform', activation='linear', weights=None,
                 border_mode='valid', subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 W_learning_rate_multiplier=None, b_learning_rate_multiplier=None,
                 input_dim=None, input_length=None, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution1D:', border_mode)
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample_length = subsample_length

        self.subsample = (subsample_length, 1)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,
                                          self.b_learning_rate_multiplier]
        
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        self.input = K.placeholder(ndim=3)
        super(Convolution1D, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.W_shape = (self.nb_filter, input_dim, self.filter_length, 1)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        length = conv_output_length(self.input_shape[1],
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0])
        return (self.input_shape[0], length, self.nb_filter)

    def get_output(self, train=False):
        X = self.get_input(train)
        X = K.expand_dims(X, -1)  # add a dimension of the right
        X = K.permute_dimensions(X, (0, 2, 1, 3))
        conv_out = K.conv2d(X, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering='th')

        output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        output = self.activation(output)
        output = K.squeeze(output, 3)  # remove the dummy 3rd dimension
        output = K.permute_dimensions(output, (0, 2, 1))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'filter_length': self.filter_length,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample_length': self.subsample_length,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'b_learning_rate_multiplier': self.b_learning_rate_multiplier,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(Convolution1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Convolution2D(Layer):
    '''Convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.


    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 W_learning_rate_multiplier=None, b_learning_rate_multiplier=None,
                 **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,\
                                          self.b_learning_rate_multiplier]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=4)
        super(Convolution2D, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        conv_out = K.conv2d(X, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.W_shape)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'b_learning_rate_multiplier': self.b_learning_rate_multiplier}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvDeconvSequence(Layer):
    '''
    '''
    input_ndim = 4

    def __init__(self, nb_filter, nb_row, nb_col,
                 pool_over_channels, pool_length, break_ties=True,
                 init='glorot_uniform', activation='sigmoid', weights=None,
                 border_mode='valid', dim_ordering='th',
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 W_learning_rate_multiplier=None, b_learning_rate_multiplier=None,
                 **kwargs):

        self.pool_over_channels=pool_over_channels
        self.pool_length = pool_length
        self.break_ties = break_ties

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = (1,1)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,\
                                          self.b_learning_rate_multiplier]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=4)
        super(ConvDeconvSequence, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            self.channel_idx = 1
            self.rows_idx = 2
            self.cols_idx = 3
            stack_size = self.input_shape[self.channel_idx]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
            self.deconv_W_shape = (self.nb_row, self.nb_filter,
                                   stack_size, self.nb_col)
        elif self.dim_ordering == 'tf':
            self.channel_idx = 3
            self.rows_idx = 1
            self.cols_idx = 2
            stack_size = self.input_shape[self.channel_idx]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
            self.deconv_W_shape = (stack_size, self.nb_col,
                                   self.nb_filter, self.nb_row)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        stack_size = self.input_shape[self.channel_idx]
        assert stack_size==1, "Preemptive error:"+\
                              " code written for 1 input channels."+\
                              " May still work but you should check."

        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        if (self.dim_ordering == 'th'):
            self.deconv_W = K.permute_dimensions(self.W,
                                         (2, #rows->outchan
                                          0, #outchan -> inchan
                                          1, #inchan -> rows
                                          3)
#reverse columns (as viewing from other side) and rows
#(which are now channels, as channels aren't subject to the reversion
#that the rows of W were subject to)
                                          )[::-1,:,:,::-1] 
        elif (self.dim_ordering=='tf'):
            self.deconv_W = K.permute_dimensions(self.W,
                                         (2, #inchan -> rows
                                          1, #columns in place
                                          3, #outchan -> inchan
                                          0) #rows -> outchan
                                          )[:,::-1,:,::-1] #reversals

        self.deconv_b = K.zeros(self.input_shape[self.rows_idx])
        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return self.input_shape

    def get_conv_out_shape(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_padding_output_shape(self, input_shape, padding):
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] + padding[0],
                    input_shape[3] + padding[1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] + padding[0],
                    input_shape[2] + padding[1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        
        #apply the conv
        conv_out = K.conv2d(X, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.W_shape)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        conv_output_shape = self.get_conv_out_shape(self.input_shape)

        #apply the activation
        conv_output = self.activation(output)

        #apply the centered pool filtering
        filtered_pool_output = MaxPoolFilter2D_CenteredPool_Sequence.\
                                get_centered_pool_output(
                                 conv_output, break_ties=self.break_ties,
                                 pool_length=self.pool_length,
                                 pool_over_channels=self.pool_over_channels,
                                 border_mode=self.border_mode,
                                 dim_ordering=self.dim_ordering,
                                 input_shape=conv_output_shape)
       
        #pad the ends so that the deconv has the right size
        padding = (0,self.nb_col-1)
        padded_filt_pool = K.spatial_2d_padding(filtered_pool_output,
                        padding=padding,
                        dim_ordering=self.dim_ordering)
        padding_output_shape = self.get_padding_output_shape(
                                    conv_output_shape, padding)

        #apply the deconv and add bias
        deconv_out = K.conv2d(padded_filt_pool, self.deconv_W,
                            strides=(1,1),
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=padding_output_shape,
                            filter_shape=self.deconv_W_shape)
        #add the bias - remember, the rows are channels for this deconv
        if self.dim_ordering == 'th':
            deconv_out = deconv_out + K.reshape(self.deconv_b,
                                          (1, self.nb_row, 1, 1))
        elif self.dim_ordering == 'tf':
            deconv_out = deconv_out + K.reshape(self.deconv_b,
                                          (1, 1, 1, self.nb_row))
        else:
            raise RuntimeError("Invalid dim ordering: "+self.dim_ordering)

        deconv_out = ExchangeChannelsAndRows.exchange_channels_and_rows(
                        X=deconv_out, dim_ordering=self.dim_ordering)

        return deconv_out

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'pool_over_channels': self.pool_over_channels,
                  'pool_length': self.pool_length,
                  'break_ties': self.break_ties,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'b_learning_rate_multiplier': self.b_learning_rate_multiplier}
        base_config = super(ConvDeconvSequence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class _Pooling1D(Layer):
    '''Abstract class for different pooling 1D layers.
    '''
    input_dim = 3

    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(_Pooling1D, self).__init__(**kwargs)
        if stride is None:
            stride = pool_length
        self.pool_length = pool_length
        self.stride = stride
        self.st = (self.stride, 1)
        self.input = K.placeholder(ndim=3)
        self.pool_size = (pool_length, 1)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode

    @property
    def output_shape(self):
        input_shape = self.input_shape
        length = conv_output_length(input_shape[1], self.pool_length,
                                    self.border_mode, self.stride)
        return (input_shape[0], length, input_shape[2])

    def _pooling_function(self, back_end, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def get_output(self, train=False):
        X = self.get_input(train)
        X = K.expand_dims(X, -1)   # add dummy last dimension
        X = K.permute_dimensions(X, (0, 2, 1, 3))
        output = self._pooling_function(inputs=X, pool_size=self.pool_size,
                                        strides=self.st,
                                        border_mode=self.border_mode,
                                        dim_ordering='th')
        output = K.permute_dimensions(output, (0, 2, 1, 3))
        return K.squeeze(output, 3)  # remove dummy last dimension

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'stride': self.stride,
                  'pool_length': self.pool_length,
                  'border_mode': self.border_mode}
        base_config = super(_Pooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling1D(_Pooling1D):
    '''Max pooling operation for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, downsampled_steps, features)`.

    # Arguments
        pool_length: factor by which to downscale. 2 will halve the input.
        stride: integer or None. Stride value.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
    '''
    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(MaxPooling1D, self).__init__(pool_length, stride,
                                           border_mode, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output


class AveragePooling1D(_Pooling1D):
    '''Average pooling for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, downsampled_steps, features)`.

    # Arguments
        pool_length: factor by which to downscale. 2 will halve the input.
        stride: integer or None. Stride value.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
    '''
    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(AveragePooling1D, self).__init__(pool_length, stride,
                                               border_mode, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='avg')
        return output


class WeightedPooling1D(_Pooling1D):
    ''' Weighted pooling for temporal data with Softmax & learned temperature parameter
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, downsampled_steps, features)`.

    # Arguments
        pool_length: factor by which to downscale. 2 will halve the input.
        stride: integer or None. Stride value.
        init: glorot_uniform (for temperature initialization) 
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
    '''
    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', init="one", **kwargs):
        super(AveragePooling1D, self).__init__(pool_length, stride,
                                               border_mode, **kwargs)
        self.init = initializations.get(init)
        assert stride==pool_lengths, 'for weighted pooling, the pool stride must equal to the pool width' 
        
    def build(self): 
        self.tau = self.init((self.input_shape[1],))
        self.trainable_weights = [self.tau]

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):

        pool_axis=-1; 
        if dim_ordering=="tf": 
            pool_axis=2; 
        #global pool (for now) 
        t_denominator=K.sum(K.exp(inputs/self.tau[None,:,None]),axis=pool_axis)
        t_softmax=K.exp(inputs/self.tau[None,:,None])/t_denominator[:,:,None]; 
        t_weighted_average=K.sum(t_softmax*inputs,axis=pool_axis)        
        output=t_weighted_average[:,:,None]; 
        return output


class _Pooling2D(Layer):
    '''Abstract class for different pooling 2D layers.
    '''
    input_ndim = 4

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.pool_size[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self._pooling_function(inputs=X, pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'pool_size': self.pool_size,
                  'border_mode': self.border_mode,
                  'strides': self.strides,
                  'dim_ordering': self.dim_ordering}
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(_Pooling2D):
    '''Max pooling operation for spatial data.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(MaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output


class AveragePooling2D(_Pooling2D):
    '''Average pooling operation for spatial data.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(AveragePooling2D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='avg')
        return output


class WeightedPooling2D(_Pooling2D):
    '''Weighted Average pooling operation for spatial data.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
        init: glorot_uniform (for temperature initialization) 
    '''
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', init="one", **kwargs):

        super(WeightedPooling2D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)
        self.init = initializations.get(init)
        assert strides==pool_size, 'for weighted pooling, the pool stride must equal to the pool width (1) ' 
        #assert strides==1, 'for weighted global average pooling, set stride=1 and stride_widths =1'
    def build(self): 
        self.tau = self.init((self.input_shape[1],))
        self.trainable_weights = [self.tau]
    
    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        pool_axis=-1; 
        if dim_ordering=="tf": 
            pool_axis=2; 
        #global pool (for now) 
        t_denominator=K.sum(K.exp(inputs/self.tau[None,:,None,None]),axis=pool_axis)
        t_softmax=K.exp(inputs/self.tau[None,:,None,None])/t_denominator[:,:,:,None]; 
        t_weighted_average=K.sum(t_softmax*inputs,axis=pool_axis)        
        output=t_weighted_average[:,:,:,None]; 
        return output


class PositionallyWeightedAveragePooling(Layer):
    '''Pooling operation that upweights hits towards the center and
    downweights hits towards the flanks.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows=1, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows=1, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, rows=1, cols=1)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows=1, cols=1, channels)` if dim_ordering='tf'.

    # Arguments
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    def __init__(self, dim_ordering='th', **kwargs):
        self.dim_ordering = dim_ordering
        super(PositionallyWeightedAveragePooling, self).__init__(**kwargs)

    def build(self): 
        if (self.dim_ordering=='th'): 
            #assuming num of rows is 1, as for sequence;
            #the positionally weighted pooling depends on dist from center
            #in the length (columns) dimension
            assert self.input_shape[2]==1, "num rows != 1"
            self.tau = K.zeros((self.input_shape[1],))
        elif (self.dim_ordering=='tf'):
            assert self.input_shape[1]==1, "num rows != 1"
            self.tau = self.zeros((self.input_shape[3],))
        self.trainable_weights = [self.tau]
    
    @property
    def output_shape(self):
        if (self.dim_ordering=='th'):
            return (self.input_shape[0], self.input_shape[1], 1, 1)
        elif (self.dim_ordering=='tf'):
            return (self.input_shape[0], 1, 1, self.input_shape[3])
        else:
            raise RuntimeError("Unsupported dim_ordering: "
                               +str(self.dim_ordering))

    def get_output(self, train=False):
        X = self.get_input(train)
        if (self.dim_ordering=='th'):
            length = X.shape[3]
        elif (self.dim_ordering=='tf'):
            length = X.shape[2]
        half_length = K.floor(length/2)

        #create a vector that represents distance from center
        dist_from_center = K.arange(0, length) 
        dist_from_center = K.set_subtensor(
            dist_from_center[0:half_length],
            dist_from_center[length-half_length:][::-1])-half_length 
        dist_from_center = dist_from_center/K.max(dist_from_center)

        #calculate the function involving tau (which goes across channels) 

        #straight line
        #dist_from_center_weights =\
        #    dist_from_center[None, :]*self.tau[:, None]

        ##power
        #dist_from_center_weights =\
        #    K.pow(dist_from_center[None,:], (self.tau[:,None]+1))

        ##temperature softmax 
        dist_from_center_times_tau = self.tau[:,None]\
                                         *dist_from_center[None,:]
        dist_from_center_weights = K.exp(dist_from_center_times_tau)

        #invert so a higher value means more downweighting
        dist_from_center_weights =\
    (K.max(dist_from_center_weights)-dist_from_center_weights)+K.epsilon() 
        #normalise
        dist_from_center_weights =\
    dist_from_center_weights/K.sum(dist_from_center_weights,axis=-1)[:,None]

        #if tensorflow, swap the axes to put the channel axis second
        if (self.dim_ordering=='th'):
            #theano dimension ordering is: sample, channel, rows, cols
            output = K.sum(X*dist_from_center_weights[None,:,None,:],
                        axis=-1, keepdims=True)
        elif (self.dim_ordering=='tf'):
            #tensorflow has channels at end, hence the need to
            #permute dimensions
            dist_from_center_weights = K.permute_dimensions(
                dist_from_center_weights, (1,0))
            output = K.sum(X*dist_from_center_weights[None,None,:,:],
                        axis=2, keepdims=True) 
        return output


class UpSampling1D(Layer):
    '''Repeat each temporal step `length` times along the time axis.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, upsampled_steps, features)`.

    # Arguments:
        length: integer. Upsampling factor.
    '''
    input_ndim = 3

    def __init__(self, length=2, **kwargs):
        super(UpSampling1D, self).__init__(**kwargs)
        self.length = length
        self.input = K.placeholder(ndim=3)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], self.length * input_shape[1], input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        output = K.repeat_elements(X, self.length, axis=1)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'length': self.length}
        base_config = super(UpSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSampling2D(Layer):
    '''Repeat the rows and columns of the data
    by size[0] and size[1] respectively.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, size=(2, 2), dim_ordering='th', **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.size = tuple(size)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    self.size[0] * input_shape[2],
                    self.size[1] * input_shape[3])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    self.size[0] * input_shape[1],
                    self.size[1] * input_shape[2],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.resize_images(X, self.size[0], self.size[1],
                               self.dim_ordering)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'size': self.size}
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPoolFilter2D_NonOverlapStrides(Layer):
    '''Only retain those positions that are the max in some maxpool kernel
       For now, only working with non-overlapping strides

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_size: tuple of 2 integers. The maxpool kernel for rows and columns
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool_size,
                 border_mode='valid', dim_ordering='th', **kwargs):
        super(MaxPoolFilter2D_NonOverlapStrides, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.pool_size = tuple(pool_size)
        self.pool_stride = self.pool_size
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        return self.input_shape

    def get_output(self, train=False):
        #only retain those positions of X that contribute to maxpool
        X = self.get_input(train)
        pool_output = K.pool2d(X, pool_size=self.pool_size,
                          strides=self.pool_size,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering, pool_mode='max')

        #upsample the pool output
        upsampled_pool_out = K.resize_images(
                                 X=pool_output,
                                 height_factor=self.pool_size[0],
                                 width_factor=self.pool_stride[1],
                                 dim_ordering=self.dim_ordering)

        if (self.dim_ordering=='th'):
            inp_rows_and_cols = [self.input_shape[2], self.input_shape[3]]
        elif (self.dim_ordering=='tf'):
            inp_rows_and_cols = [self.input_shape[1], self.input_shape[2]]
        upsample_pool_size = tuple(
            [(int((inp_rows_and_cols[i]-
                  self.pool_size[i])/self.pool_stride[i])+1)*self.pool_size[i]
             for i in range(2)])

        #pad on right to be same dimensions as input
        upsampled_pool_out_padded = K.zeros_like(X) 
        if (self.dim_ordering=='th'):
            subtensor = upsampled_pool_out_padded[:,:,
                            :upsample_pool_size[0],
                            :upsample_pool_size[1]]
        else:
            subtensor =  upsampled_pool_out_padded[:,
                            :upsample_pool_size[0],
                            :upsample_pool_size[1],:]
        upsampled_pool_out_padded = K.set_subtensor(
                                      subtensor,
                                      upsampled_pool_out)

        #only return those positions in the input that are
        #equal to the padded upsampled pooled output 
        output = K.switch(K.equal(upsampled_pool_out_padded,X), X, 0) 
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'pool_size': self.pool_size,
                  'border_mode': self.border_mode}
        base_config = super(MaxPoolFilter2D_NonOverlapStrides,
                            self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPoolFilter2D_CenteredPool_Sequence(Layer):
    '''Only retain those positions that are the max in a pool window
    centered on the region. Only works with stride 1. For even
    windows, the "center" is taken as the neuron to the left.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_length: length of pooling 
        break_ties: Break ties using a small amount of random noise
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool_length, pool_over_channels, break_ties=True,
                 border_mode='valid', dim_ordering='th', **kwargs):
        super(MaxPoolFilter2D_CenteredPool_Sequence, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.break_ties = break_ties
        self.pool_length = pool_length
        self.pool_over_channels = pool_over_channels
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        return self.input_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.get_centered_pool_output(X=X, break_ties=self.break_ties,
                        pool_length=self.pool_length,
                        pool_over_channels=self.pool_over_channels,
                        border_mode=self.border_mode,
                        dim_ordering=self.dim_ordering,
                        input_shape=self.input_shape)

    @staticmethod
    def get_centered_pool_output(X, break_ties, pool_length,
                                    pool_over_channels, border_mode,
                                    dim_ordering, input_shape):
        #Determine pooling settings and infer shape of pooling output
        if (dim_ordering=='th'):
            #sanity check that with sequence data - rows are len 1
            assert input_shape[2]==1
            inp_rows_and_cols = [input_shape[2], input_shape[3]]
            num_channels = input_shape[1]
        elif (dim_ordering=='tf'):
            #sanity check that with sequence data - rows are len 1
            assert input_shape[1]==1
            inp_rows_and_cols = [input_shape[1], input_shape[2]]
            num_channels = input_shape[3]

        if (pool_over_channels):
            if (dim_ordering=='th'): 
                swap_channels_and_rows_order =  (0,2,1,3)
            else:
                swap_channels_and_rows_order = (0,3,2,1)
            #swap channels and rows
            X = K.permute_dimensions(X, swap_channels_and_rows_order) 
            pool_size = (num_channels, pool_length)
        else:
            pool_size = (1, pool_length)
        pool_output_size = (1, (inp_rows_and_cols[1]-pool_length+1))

        #break ties if break_ties is True
        if (break_ties):
            X_tiebreak = X + K.random_uniform(X.shape, high=10**-6)
        else:
            X_tiebreak = X

        #do a maxpool with stride 1
        pool_output = K.pool2d(X_tiebreak, pool_size=pool_size,
                          strides=(1,1),
                          border_mode=border_mode,
                          dim_ordering=dim_ordering, pool_mode='max')

        #determine the padding for the maxpool
        left_pad = int((pool_length-1)/2)
        right_pad = (pool_length-1) - left_pad

        #pad pooling output to have same length as input
        #(will broadcast along the rows dimension)
        if (dim_ordering=='th'):
            pool_out_padded = K.zeros_shape_is_variable(
                                (X_tiebreak.shape[0],
                                 X_tiebreak.shape[1], 1,
                                 X_tiebreak.shape[3])) 
            subtensor = pool_out_padded[:,:,:,
                         left_pad:inp_rows_and_cols[1]-right_pad]
        elif (dim_ordering=='tf'):
            pool_out_padded = K.zeros_shape_is_variable(
                                (X_tiebreak.shape[0],
                                 1, X_tiebreak.shape[2], X_tiebreak.shape[3])) 
            subtensor =  pool_out_padded[:,:,
                          left_pad:inp_rows_and_cols[1]-right_pad,:]
        pool_out_padded = K.set_subtensor(subtensor, pool_output)

        #only return those positions in the input that are
        #equal to the padded pooled output, which will only happen
        #when those positions are the max of a pool_output_size window
        #centered on them.
        output = K.switch(K.equal(pool_out_padded,X_tiebreak), X, 0) 

        if (pool_over_channels):
            #swap axes back
            output = K.permute_dimensions(output, swap_channels_and_rows_order) 

        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'pool_length': self.pool_length,
                  'pool_over_channels': self.pool_over_channels,
                  'break_ties': self.break_ties,
                  'border_mode': self.border_mode}
        base_config = super(MaxPoolFilter2D_CenteredPool_Sequence,
                            self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftmaxAcrossAxis(Layer):
    '''Applies a softmax operation across a specified axis

    # Input shape
        anything

    # Output shape
        same as input shape
    '''

    def __init__(self, axis, **kwargs):
        super(SoftmaxAcrossAxis, self).__init__(**kwargs)
        self.axis = axis
        self.input = K.placeholder(ndim=4)

    @property
    def output_shape(self):
        return self.input_shape

    def get_output(self, train=False):
        new_axis_order = [i for i in range(4) if i != self.axis]
        new_axis_order.append(self.axis)
        inverse_axis_order = [None]*4
        for (i,j) in enumerate(new_axis_order):
            inverse_axis_order[j] = i
        
        X = self.get_input(train)
        original_X_shape = X.shape
        #permute
        X = K.permute_dimensions(X, new_axis_order) 
        #reshape
        permuted_X_shape = X.shape
        reshape_axis_1 = (permuted_X_shape[0]*permuted_X_shape[1]
                          *permuted_X_shape[2])
        X = K.reshape(X, (reshape_axis_1, permuted_X_shape[3])) 
        #softmax
        softmaxed_X = K.softmax(X)
        #unreshape
        softmaxed_X = K.reshape(softmaxed_X, permuted_X_shape)
        #unpermute
        softmaxed_X = K.permute_dimensions(softmaxed_X, inverse_axis_order)
        return softmaxed_X


class SoftmaxAcrossRows(SoftmaxAcrossAxis):
    '''Applies a softmax operation across the rows

    # Input shape
        anything

    # Output shape
        same as input shape
    '''

    def __init__(self, dim_ordering='th', **kwargs):
        if (dim_ordering=='th'):
            axis=2
        elif (dim_ordering=='tf'):
            axis=1
        super(SoftmaxAcrossRows, self).__init__(axis=axis, **kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(SoftmaxAcrossRows, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ExchangeChannelsAndRows(Layer):
    '''Exchanges the channels and rows axes

    # Input shape
        anything

    # Output shape
        same as input but with channel and row dims switched
    '''

    def __init__(self, dim_ordering='th', **kwargs):
        super(ExchangeChannelsAndRows, self).__init__(**kwargs)
        self.dim_ordering=dim_ordering

    @property
    def output_shape(self):
        if (self.dim_ordering=='th'):
            return tuple([self.input_shape[x] for x in self.th_permute])
        elif (self.dim_ordering=='tf'):
            return tuple([self.input_shape[x] for x in self.tf_permute])

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.exchange_channels_and_rows(X=X,
                        dim_ordering=self.dim_ordering)

    @staticmethod
    def exchange_channels_and_rows(X, dim_ordering):
        if (dim_ordering=='th'):
            new_order = (0,2,1,3)
        elif (dim_ordering=='tf'):
            new_order = (0,3,2,1)
        return K.permute_dimensions(X, new_order) 


class ZeroPadding1D(Layer):
    '''Zero-padding layer for 1D input (e.g. temporal sequence).

    # Input shape
        3D tensor with shape (samples, axis_to_pad, features)

    # Output shape
        3D tensor with shape (samples, padded_axis, features)

    # Arguments
        padding: int
            How many zeros to add at the beginning and end of
            the padding dimension (axis 1).
    '''
    input_ndim = 3

    def __init__(self, padding=1, **kwargs):
        super(ZeroPadding1D, self).__init__(**kwargs)
        self.padding = padding
        self.input = K.placeholder(ndim=3)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0],
                input_shape[1] + self.padding * 2,
                input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.temporal_padding(X, padding=self.padding)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'padding': self.padding}
        base_config = super(ZeroPadding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding2D(Layer):
    '''Zero-padding layer for 2D input (e.g. picture).

    # Input shape
        4D tensor with shape:
        (samples, depth, first_axis_to_pad, second_axis_to_pad)

    # Output shape
        4D tensor with shape:
        (samples, depth, first_padded_axis, second_padded_axis)

    # Arguments
        padding: tuple of int (length 2)
            How many zeros to add at the beginning and end of
            the 2 padding dimensions (axis 3 and 4).
    '''
    input_ndim = 4

    def __init__(self, padding=(1, 1), dim_ordering='th', **kwargs):
        super(ZeroPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input = K.placeholder(ndim=4)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2 * self.padding[0],
                    input_shape[3] + 2 * self.padding[1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] + 2 * self.padding[0],
                    input_shape[2] + 2 * self.padding[1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.spatial_2d_padding(X, padding=self.padding,
                                    dim_ordering=self.dim_ordering)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'padding': self.padding}
        base_config = super(ZeroPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

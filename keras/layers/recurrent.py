# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_scalar, shared_zeros, shared_ones, alloc_zeros_matrix
from ..layers.core import Layer, MaskedLayer
from six.moves import range


class Recurrent(MaskedLayer):
    def get_output_mask(self, train=None):
        if self.return_sequences:
            return super(Recurrent, self).get_output_mask(train)
        else:
            return None

    def get_padded_shuffled_mask(self, train, X, pad=0):
        mask = self.get_input_mask(train)
        if mask is None:
            mask = T.ones_like(X.sum(axis=-1))  # is there a better way to do this without a sum?

        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # the new dimension (the '1') is made broadcastable
        # see http://deeplearning.net/software/theano/library/tensor/basic.html#broadcasting-in-theano-vs-numpy
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')


class SimpleRNN(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False):

        super(SimpleRNN, self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, mask_tm1, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        return self.activation(x_t + mask_tm1 * T.dot(h_tm1, u))

    def get_output(self, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=self.U,  # static inputs to _step
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class SimpleDeepRNN(Recurrent):
    '''
        Fully connected RNN where the output of multiple timesteps
        (up to "depth" steps in the past) is fed back to the input:

        output = activation( W.x_t + b + inner_activation(U_1.h_tm1) + inner_activation(U_2.h_tm2) + ... )

        This demonstrates how to build RNNs with arbitrary lookback.
        Also (probably) not a super useful model.
    '''
    def __init__(self, input_dim, output_dim, depth=3,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(SimpleDeepRNN, self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.depth = depth
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.Us = [self.inner_init((self.output_dim, self.output_dim)) for _ in range(self.depth)]
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W] + self.Us + [self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, *args):
        o = x_t
        for i in range(self.depth):
            mask_tmi = args[i]
            h_tmi = args[i + self.depth]
            U_tmi = args[i + 2*self.depth]
            o += mask_tmi*self.inner_activation(T.dot(h_tmi, U_tmi))
        return self.activation(o)

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=self.depth)
        X = X.dimshuffle((1, 0, 2))

        x = T.dot(X, self.W) + self.b

        if self.depth == 1:
            initial = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        else:
            initial = T.unbroadcast(T.unbroadcast(alloc_zeros_matrix(self.depth, X.shape[1], self.output_dim), 0), 2)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x, dict(
                input=padded_mask,
                taps=[(-i) for i in range(self.depth)]
            )],
            outputs_info=[dict(
                initial=initial,
                taps=[(-i-1) for i in range(self.depth)]
            )],
            non_sequences=self.Us,
            truncate_gradient=self.truncate_gradient
        )

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "depth": self.depth,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}




class Counter(Recurrent):
    '''
        My attempt at constructing counter units

        incrementAmount controls how much the counter unit is
        incremented by when the start gate is active. incrementAmountInit
        is the val to init this too...
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 inner_activation='hard_sigmoid', #huh...hard_sigmoid is faster than sigmoid but do I want it?
                 weights=None, truncate_gradient=-1, return_sequences=False, incrementAmountInit=0.2, learnIncrementAmount=False):

        super(Counter, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        #interpret z as the "keep counting"
        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        #interpret r as the stopwatch reset gate
        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        #incrementAmount is the amount to increment by when the start gate is active
        self.incrementAmount = shared_scalar(val=incrementAmountInit); 
        self.learnIncrementAmount = learnIncrementAmount;        

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
        ]
        if (self.learnIncrementAmount):
            self.params.append(self.incrementAmount);

        if weights is not None: #what does this do?? What weights??
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, mask_tm1,
              h_tm1,
              u_z, u_r):
        h_mask_tm1 = mask_tm1 * h_tm1 #for dropout?
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))

        h_t = z*self.incrementAmount + h_tm1*(1-r); #using h_tm1 and not h_mask_tm1 because otherwise acts like a "reset" flip...(assuming masking is for dropout...)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1) #for dropout?
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}

class EventSpacingCounter_Thresholds(Recurrent):
    '''
        I am going to engineer these to specifically detect the spacing between two events...
        This unit here is super minimal
    '''
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 startStopActivation='tanh', inner_activation='tanh', incrementAmount=0.2,
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(EventSpacingCounter_Thresholds, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim #countOn, countHiddenState
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.incrementAmount = incrementAmount;

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.startStopActivation = activations.get(startStopActivation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_startEvent = self.init((self.input_dim, self.output_dim));
        self.b_startEvent = shared_zeros((self.output_dim));

        self.W_stopEvent = self.init((self.input_dim, self.output_dim));
        self.b_stopEvent = shared_zeros((self.output_dim));

        self.thresholds1 = shared_zeros((self.output_dim));
        self.thresholds1Scale = shared_ones((self.output_dim));   

        self.params = [
            self.W_startEvent, self.b_startEvent
            , self.W_stopEvent, self.b_stopEvent
            , self.thresholds1, self.thresholds1Scale
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              startEvent_t, stopEvent_t, mask_tm1,
              h_counterStates_tm1, h_counterValues_tm1, h_counterValTestsPassed_tm1
              ):
        #I am going to ignore the mask...I don't
        #see this playing well with dropout at all...
        #h_mask_tm1 = mask_tm1 * h_tm1
        
        h_counterStates = self.inner_activation(5*(h_counterStates_tm1 + 2*((1-h_counterStates_tm1)*startEvent_t - (h_counterValues_tm1)*stopEvent_t)));
        h_counterValues = h_counterStates*self.incrementAmount + h_counterValues_tm1;
       
        #ugh are we layering too many tanh's here?
        h_counterValTestsPassed = self.inner_activation(self.thresholds1Scale*(h_counterValues-self.thresholds1));

        return h_counterStates, h_counterValues, h_counterValTestsPassed;

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2)) #batch axis first

        startEvent = self.startStopActivation(T.dot(X, self.W_startEvent) + self.b_startEvent)
        stopEvent = self.startStopActivation(T.dot(X, self.W_stopEvent) + self.b_stopEvent)
        [counterStates,counterValues,outputs], updates = theano.scan(
            self._step,
            sequences=[startEvent, stopEvent, padded_mask],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                          T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                          T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            non_sequences=[],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "startStopActivation": self.startStopActivation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences,
                "incrementAmount": self.incrementAmount}

class EventSpacingCounter_Minimal(Recurrent):
    '''
        I am going to engineer these to specifically detect the spacing between two events...
        This unit here is super minimal
    '''
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 startStopActivation='tanh', inner_activation='tanh', incrementAmount=0.2,
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(EventSpacingCounter_Minimal, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim #countOn, countHiddenState
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.incrementAmount = incrementAmount;

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.startStopActivation = activations.get(startStopActivation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_startEvent = self.init((self.input_dim, self.output_dim));
        self.b_startEvent = shared_zeros((self.output_dim));

        self.W_stopEvent = self.init((self.input_dim, self.output_dim));
        self.b_stopEvent = shared_zeros((self.output_dim));

        self.params = [
            self.W_startEvent, self.b_startEvent
            , self.W_stopEvent, self.b_stopEvent
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              startEvent_t, stopEvent_t, mask_tm1,
              h_counterStates_tm1, h_counterValues_tm1,
              ):
        #I am going to ignore the mask...I don't
        #see this playing well with dropout at all...
        #h_mask_tm1 = mask_tm1 * h_tm1
        
        h_counterStates = self.inner_activation(5*(h_counterStates_tm1 + 2*((1-h_counterStates_tm1)*startEvent_t - (h_counterValues_tm1)*stopEvent_t)));
        h_counterValues = h_counterStates*self.incrementAmount + h_counterValues_tm1;
        return h_counterStates, h_counterValues;

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2)) #batch axis first

        startEvent = self.startStopActivation(T.dot(X, self.W_startEvent) + self.b_startEvent)
        stopEvent = self.startStopActivation(T.dot(X, self.W_stopEvent) + self.b_stopEvent)
        [counterStates,outputs], updates = theano.scan(
            self._step,
            sequences=[startEvent, stopEvent, padded_mask],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                          T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            non_sequences=[],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "startStopActivation": self.startStopActivation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences,
                "incrementAmount": self.incrementAmount}

class CounterGRU(Recurrent):
    '''
        I want this to contain a mix of Counter units and the GRU units
    
        Ugh...should restructure this to be like the LSTM so it will
        execute faster. The slicing is a sloww operation.
    '''
    def __init__(self, input_dim, num_gru_outputs, num_counter_outputs,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False, incrementAmountInit=0.2, learnIncrementAmount=False):

        super(CounterGRU, self).__init__()
        self.input_dim = input_dim
        self.num_gru_outputs = num_gru_outputs
        self.num_counter_outputs = num_counter_outputs
        self.output_dim = (num_gru_outputs + num_counter_outputs)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_z_gru = self.init((self.input_dim, self.num_gru_outputs))
        self.U_z_gru = self.inner_init((self.output_dim, self.num_gru_outputs))
        self.b_z_gru = shared_zeros((self.num_gru_outputs))

        #The "reset gate" for a GRU is "how much of previous hidden state is exposed to
        #self.U_h_gru...so that is why the reset has to apply to gruAndCounter
        self.W_r_gruAndCounter = self.init((self.input_dim, self.output_dim))
        self.U_r_gruAndCounter = self.inner_init((self.output_dim, self.output_dim))
        self.b_r_gruAndCounter = shared_zeros((self.output_dim))

        self.W_h_gru = self.init((self.input_dim, self.num_gru_outputs))
        self.U_h_gru = self.inner_init((self.output_dim, self.num_gru_outputs))
        self.b_h_gru = shared_zeros((self.num_gru_outputs))

        self.W_z_counter = self.init((self.input_dim, self.num_counter_outputs))
        self.U_z_counter = self.inner_init((self.output_dim, self.num_counter_outputs))
        self.b_z_counter = shared_zeros((self.num_counter_outputs))

        self.W_r_counter = self.init((self.input_dim, self.num_counter_outputs))
        self.U_r_counter = self.inner_init((self.output_dim, self.num_counter_outputs))
        self.b_r_counter = shared_zeros((self.num_counter_outputs))

        self.incrementAmount = shared_scalar(val=incrementAmountInit); 

        self.params = [
            self.W_z_gru, self.U_z_gru, self.b_z_gru,
            self.W_r_gruAndCounter, self.U_r_gruAndCounter, self.b_r_gruAndCounter,
            self.W_h_gru, self.U_h_gru, self.b_h_gru,
            self.W_z_counter, self.U_z_counter, self.b_z_counter,
            self.W_r_counter, self.U_r_counter, self.b_r_counter,
        ]
        #if (learnIncrementAmount):
        #    self.params.append(self.incrementAmount);

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t_gru, xr_t_gruAndCounter, xh_t_gru, xz_t_counter, xr_t_counter, mask_tm1,
              h_tm1,
              u_z_gru, u_r_gruAndCounter, u_h_gru, u_z_counter, u_r_counter):
        h_mask_tm1 = mask_tm1 * h_tm1

        #notice that the z/r gates of the gru/counter compute their
        #activation using BOTH the gru and counter units of the
        #hidden state
        z_gru = self.inner_activation(xz_t_gru + T.dot(h_mask_tm1, u_z_gru))
        r_gruAndCounter = self.inner_activation(xr_t_gruAndCounter + T.dot(h_mask_tm1, u_r_gruAndCounter))
        h_mask_tm1_gru = h_mask_tm1[:,:self.num_gru_outputs]; 
        hh_t_gru = self.activation((xh_t_gru) + T.dot(r_gruAndCounter * h_mask_tm1, u_h_gru))
        h_t_gru = z_gru * h_mask_tm1_gru + (1 - z_gru) * hh_t_gru

        z_counter = self.inner_activation(xz_t_counter + T.dot(h_mask_tm1, u_z_counter))
        #z_counter = (xz_t_counter + T.dot(h_mask_tm1, u_z_counter))
        r_counter = self.inner_activation(xr_t_counter + T.dot(h_mask_tm1, u_r_counter))
        #h_mask_tm1 = h_mask_tm1+T.zeros_like(u_z_gru);#like a die command. Crash at right spot.

        h_tm1_counter = h_tm1[:,self.num_gru_outputs:];
        #h_t_counter = z_counter*self.incrementAmount  + h_tm1_counter*(1-r_counter), -5,2; #using h_tm1 and not h_mask_tm1 because otherwise acts like a "reset" flip...(assuming masking is for dropout...)
        h_t_counter = z_counter + h_tm1_counter*(1-r_counter); #using h_tm1 and not h_mask_tm1 because otherwise acts like a "reset" flip...(assuming masking is for dropout...)
        #toReturn = T.zeros_like(h_tm1);
        #toReturn[:,:self.num_gru_outputs] = h_t_gru;
        #toReturn[:,self.num_gru_outputs:] = h_t_counter;
        #return toReturn;
        return T.concatenate([h_t_gru, h_t_counter],axis=1);

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2)) #batch axis first

        x_z_gru = T.dot(X, self.W_z_gru) + self.b_z_gru
        x_r_gru = T.dot(X, self.W_r_gruAndCounter) + self.b_r_gruAndCounter
        x_h_gru = T.dot(X, self.W_h_gru) + self.b_h_gru
        
        x_z_counter = T.dot(X, self.W_z_counter) + self.b_z_counter
        x_r_counter = T.dot(X, self.W_r_counter) + self.b_r_counter
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z_gru, x_r_gru, x_h_gru, x_z_counter, x_r_counter, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z_gru, self.U_r_gruAndCounter, self.U_h_gru, self.U_z_counter, self.U_r_counter],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}

class GRU(Recurrent):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class LSTM(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class JZS1(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT1` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(JZS1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if self.input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P = np.random.binomial(1, 0.5, size=(self.input_dim, self.output_dim)).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(self.input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.U_h, self.b_h,
            self.Pmat
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t)
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.tanh(T.dot(X, self.Pmat)) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class JZS2(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT2` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(JZS2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if self.input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P = np.random.binomial(1, 0.5, size=(self.input_dim, self.output_dim)).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(self.input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.Pmat
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.Pmat) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class JZS3(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT3` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(JZS3, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(T.tanh(h_mask_tm1), u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}

from __future__ import absolute_import
import numpy as np
from . import backend as K
#from theano import printing 
import pdb 


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

#This may or may not be a useful function:
#does it make sense to have task weights for regression? 
def get_weighted_mean_squared_error(w0_weights,w1_weights,thresh=0.5): 
    w0_weights=np.array(w0_weights); 
    w1_weights=np.array(w1_weights);
    def weighted_mean_squared_error(y_true,y_pred): 
        y_true_binarized=(y_true<thresh).astype(int) 
        weightVectors = y_true_binarized*w1_weights[None,:] + (1-y_true_binarized)*w0_weights[None,:] 
        return K.mean(K.square(y_pred-y_true)*weightVectors, axis=-1);
    return weighted_mean_squared_error; 
        
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    nonAmbig=(y_true > -.5).nonzero() 
    #print(K.eval(y_true))
    #print(len(K.eval(nonAmbig[0])), len(K.eval(nonAmbig[1])))
    #print(K.eval(y_pred[nonAmbig]))
    #print(K.eval(y_true[nonAmbig]))
    #print("out",K.eval(K.categorical_crossentropy(y_pred[nonAmbig], y_true[nonAmbig])))
    return K.categorical_crossentropy(y_pred[nonAmbig], y_true[nonAmbig])
    

def binary_crossentropy(y_true, y_pred):
    nonAmbig=(y_true > -0.5).nonzero()  
    return K.mean(K.binary_crossentropy(y_pred[nonAmbig], y_true[nonAmbig]), axis=-1)


def get_weighted_binary_crossentropy(w0_weights, w1_weights):
    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
    # In addition, weight everything with label -1 to 0
    w0_weights=np.array(w0_weights);
    w1_weights=np.array(w1_weights);
    def weighted_binary_crossentropy(y_true,y_pred): 
        weightsPerTaskRep = y_true*w1_weights[None,:] + (1-y_true)*w0_weights[None,:]
        nonAmbig = (y_true > -0.5)
        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
        return K.mean(K.binary_crossentropy(y_pred, y_true)*nonAmbigTimesWeightsPerTask, axis=-1);
    return weighted_binary_crossentropy; 


def one_hot_rows_categorical_cross_entropy(y_true, y_pred):
    '''expects y_true and y_pred to be of dims
    samples x 1 x one-hot rows x length (if theano dim ordering) or
    samples x one-hot rows x length x 1 (if tensorflow dim ordering).
    WILL APPLY THE SOFTMAX ITSELF'''
    if K._BACKEND=='theano':
        y_true = y_true[:,0,:,:]
        y_pred = y_pred[:,0,:,:]
    elif K._BACKEND=='tensorflow':
        y_true = y_true[:,:,:,0]
        y_pred = y_pred[:,:,:,0]
    samples = y_true.shape[0]
    rows = y_true.shape[1]
    length = y_true.shape[2]

    #permute to samples x length x one-hot-rows
    y_true = K.permute_dimensions(y_true, (0,2,1)) 
    y_pred = K.permute_dimensions(y_pred, (0,2,1)) 
    #reshape to (samples*length) x one-hot-rows
    y_true = K.reshape(y_true, (samples*length, rows))
    y_pred = K.reshape(y_pred, (samples*length, rows))
    #apply categorical cross-entropy 
    loss = K.categorical_crossentropy(y_pred, y_true, from_logits=False)
    #reshape loss from (samples*length) to samples x length
    loss = K.reshape(loss, (samples, length)) 
    #take mean across length to return something of size samples
    return K.mean(loss, axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)




# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity
from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

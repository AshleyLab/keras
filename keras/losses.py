from __future__ import absolute_import
import six
from . import backend as K
from .utils.generic_utils import deserialize_keras_object
from .utils.conv_utils import normalize_padding 
import numpy as np 
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def logcosh(y_true, y_pred):
    def cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    return K.mean(K.log(cosh(y_pred - y_true)), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true)


def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)




#KUNDAJE LAB FUNCTIONS
def get_positionwise_cosine_1d(pool_size,
                               non_zero_penalty,
                               strides=1,
                               padding='same',
                               data_format='channels_last',
                               pseudocount=0.00001):
    padding=normalize_padding(padding) 
    def do_sum_pool_1d(input_to_pool):
        #sum channels, replace w/ dummy
        input_to_pool = K.expand_dims(K.sum(input_to_pool,axis=-1),axis=2)
        #sum lengthwise
        return K.squeeze(K.squeeze(K.pool2d(K.expand_dims(input_to_pool,2),
                          pool_size=(pool_size,1),
                          strides=(strides,1),
                          padding=padding,
                          data_format=data_format,
                          pool_mode='avg')*pool_size,axis=2),axis=2)  
    def positionwise_cosine_1d(y_true,y_pred):
        y_true=y_true-K.expand_dims(K.mean(y_true,axis=2),axis=2)
        y_pred = y_pred-K.expand_dims(K.mean(y_pred,axis=2),axis=2)  
        mask_y_true= K.cast(K.greater(K.abs(y_true),0),'float32')
        sum_pool_mask=do_sum_pool_1d(mask_y_true)
        max_pool_sum_pool_mask = K.squeeze(K.squeeze(K.pool2d(
            K.expand_dims(K.expand_dims(sum_pool_mask,2),2), 
                          pool_size=(pool_size,1),                              
                          strides=(strides,1),                                  
                          padding=padding,                                      
                          data_format=data_format,                              
                          pool_mode='max'),axis=2),axis=2)
        nonoverlap_mask=K.cast(K.equal(max_pool_sum_pool_mask,sum_pool_mask),'float32')
        elemwise_prod=y_true*y_pred
        pooled_elemwise_prod=do_sum_pool_1d(elemwise_prod)*nonoverlap_mask
        y_true_mag=K.sqrt(do_sum_pool_1d(y_true*y_true)+pseudocount)
        y_pred_mag=K.sqrt(K.abs(do_sum_pool_1d(y_pred*y_pred))+pseudocount)

        positive_loss = K.sum(-(pooled_elemwise_prod)/(y_true_mag*y_pred_mag+pseudocount),axis=1)
        negative_loss = K.sum(K.abs((1-mask_y_true)*(y_pred))*non_zero_penalty)

        return positive_loss + negative_loss
    return positionwise_cosine_1d
    

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

def get_weighted_binary_crossentropy(w0_weights, w1_weights):
    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
    # In addition, weight everything with label -1 to 0
    w0_weights=np.array(w0_weights);
    w1_weights=np.array(w1_weights);
    def weighted_binary_crossentropy(y_true,y_pred):
        weightsPerTaskRep = y_true*w1_weights[None,:] + (1-y_true)*w0_weights[None,:]
        nonAmbig = K.cast((y_true > -0.5),'float32')
        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
        return K.mean(K.binary_crossentropy(y_pred, y_true)*nonAmbigTimesWeightsPerTask, axis=-1);
    return weighted_binary_crossentropy; 


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


def serialize(loss):
    return loss.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)

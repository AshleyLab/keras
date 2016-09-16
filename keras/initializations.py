from __future__ import absolute_import
import numpy as np
from . import backend as K


def get_fans(shape, dim_ordering='th'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def uniform(shape, scale=0.05, name=None):
    return K.random_uniform_variable(shape, -scale, scale, name=name)

def normal(shape, scale=0.05, name=None):
    return K.random_normal_variable(shape, 0.0, scale, name=name)

def nMers(shape, n, name=None):
    import itertools;
    assert len(shape)==4, "expecting axes: outChannel, inChannel, rows, cols but got shape: "+str(shape);
    assert shape[1]==1, "expecting inChannel to be of len 1 because of on-hot encoding but got: "+str(shape[1])
    assert shape[2]==4, "expecting 3rd axis to be of len 4 because rows for DNA seq but has len "+str(shape[2]);
    assert shape[3]==n, "expecting 4th axis to be n="+str(n)+" because it's the width but is "+str(shape[3])
  
    weights = np.zeros(shape); 
    #compute n^4 + (n-1)^4 + ... 
    numChannels = sum([4**x for x in range(1,n+1)]);
    assert shape[0]==numChannels, "must supply numFilters="+str(numChannels)+" for n="+str(n)+" (you gave "+str(shape[0])+")";
    #get all possible 4-mers
    filterIdx=0;
    for k in range(1,n+1):
        allKmers = list(itertools.product([0,1,2,3],repeat=k));
        offsetOfKmerFromStart = int((n-k)/2); 
        weights[filterIdx:filterIdx+len(allKmers),0,:,offsetOfKmerFromStart:offsetOfKmerFromStart+k] = -(k-1);
        #print("initialising",len(list(allKmers)),"kmers of len",k);
        for kmerArr in allKmers:
            #kmerArr is an array of length k where entries are 0/1/2/3 representing the bases
            #do fancy indexing into weights to set the appropriate positions to 1
            weights[filterIdx
                    ,0       
                    ,kmerArr #A/C/G/T
                    ,range(offsetOfKmerFromStart,offsetOfKmerFromStart+k)] = 1.0
            filterIdx+=1; 
    assert filterIdx==numChannels,str(filterIdx)
    return K.variable(weights,name=name);

def threeMers(shape, name=None):
    return nMers(shape, 3, name=name);

def fourMers(shape, name=None):
    return nMers(shape, 3, name=name);

def fiveMers(shape, name=None):
    return nMers(shape, 5, name=name);

def lecun_uniform(shape, name=None, dim_ordering='th'):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale, name=name)


def glorot_normal(shape, name=None, dim_ordering='th'):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s, name=name)


def glorot_uniform(shape, name=None, dim_ordering='th'):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)


def he_normal(shape, name=None, dim_ordering='th'):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s, name=name)


def he_uniform(shape, name=None, dim_ordering='th'):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s, name=name)


def orthogonal(shape, scale=1.1, name=None):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return K.variable(scale * q[:shape[0], :shape[1]], name=name)


def identity(shape, scale=1, name=None):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Exception('Identity matrix initialization can only be used '
                        'for 2D square matrices.')
    else:
        return K.variable(scale * np.identity(shape[0]), name=name)


def zero(shape, name=None):
    return K.zeros(shape, name=name)


def one(shape, name=None):
    return K.ones(shape, name=name)


from .utils.generic_utils import get_from_module
def get(identifier, **kwargs):
    return get_from_module(identifier, globals(),
                           'initialization', kwargs=kwargs)

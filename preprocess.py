#This module performs mean subtraction and normalization
import numpy as np 
import pdb 

def mean_subtract(X):
    X=np.asarray(X) 
    X -= (np.mean(X, axis = 0))
    return X

def normalize(X):
    X=(np.asarray(X)).astype(np.float64) 
    X /= np.std(X, axis = 0)
    return X

def online_variance(data):
    n = 0
    mean = 0
    M2 = 0

    for x in data:
        n = n + 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)

    variance = M2/(n - 1)
    return variance



def preprocess(X_train,X_valid,X_test,nonzeroindices_train,nonzeroindices_valid):
    #mean_train=np.mean(X_train[nonzeroindices_train],axis=0)
    num_train=X_train.shape[0] 
    n=0 
    mean=0 
    M2=0 
    for i in range(num_train): 
        n=n+1
        delta=X_train[i]-mean 
        mean=mean+delta*1.0/n
        M2=M2+delta*(X_train[i]-mean)
    std_train=np.sqrt(M2/(n-1))
    #std_train=np.std(X_train[nonzeroindices_train],axis=0) 
    np.save('std.train',std_train) 
    #np.save('mean.train',mean_train) 
    #pdb.set_trace() 
    mean_train=np.load('mean.train.pkl') 
    std_train=np.load('std.train.pkl') 
    nonzeroindices_train=np.asarray(nonzeroindices_train).tolist()[0] 
    nonzeroindices_valid=np.asarray(nonzeroindices_valid).tolist()[0]  
    #pdb.set_trace() 
    for i in nonzeroindices_train:  
        X_train[i]=np.divide((X_train[i]-mean_train),std_train)
        if i%1000==0: 
            print str(i) 
    num_valid=X_valid.shape[0] 
    for i in nonzeroindices_valid:
        X_valid[i]=np.divide((X_valid[i]-mean_train),std_train) 
    X_test=(X_test-mean_train)/std_train 

    #X_train=normalize(mean_subtract(X_train))
    #X_valid=normalize(mean_subtract(X_valid))
    #X_test=normalize(mean_subtract(X_test))
    return X_train,X_valid,X_test


from Params import * 
from load_data import *
from preprocess import *
from vgg16_keras import * 
import h5py 

import keras
import theano
#for visualization:
#from keras.utils.visualize_util import plot
from keras.optimizers import nadam  #SGD, Adagrad, Adadelta, Adam, nadam 

import pickle
import numpy as np 
import sys
import pdb 

#use generator to fit the model! 
def create_generator(samplesToYield,inputFile):
    x_key='X_train'
    y_key='Y_train'
    numEntries=inputFile[x_key].shape[0] 
    while 1: 
        batchIndex=random.randint(0,numEntries-samplesToYield) 
        X=np.asarray(inputFile[x_key][batchIndex:batchIndex+samplesToYield])
        Y=np.asarray(inputFile[y_key][batchIndex:batchIndex+samplesToYield])
        yield X,Y


def main():
    #load data 
    #X_train,Y_train,X_valid,Y_valid,X_test,last_label_index=load_data(training_dir,valid_dir,test_dir,labels,sample)
    '''
    f=h5py.File('imagenet.with.zeros.hdf5','r') 
    X_train=np.asarray(f['X_train'])
    Y_train=np.asarray(f['Y_train']) 
    X_valid=np.asarray(f['X_valid']) 
    Y_valid=np.asarray(f['Y_valid']) 
    X_test=np.asarray(f['X_test']) 
    print(str(Y_train.shape))
    nonzeroindices_train=np.where(Y_train[:,0]==0)
    print str(nonzeroindices_train) 
    nonzeroindices_valid=np.where(Y_valid[:,0]==0)
    #preprocess data by mean subtraction and normalization 
    X_train,X_valid,X_test=preprocess(X_train,X_valid,X_test,nonzeroindices_train,nonzeroindices_valid)
    #pdb.set_trace() 
    f=h5py.File('imagenet.normalized.with.zeros.hdf5','w')
    dset_xtrain=f.create_dataset("X_train",data=X_train)
    dset_ytrain=f.create_dataset("Y_train",data=Y_train) 
    dset_xvalid=f.create_dataset("X_valid",data=X_valid) 
    dset_yvalid=f.create_dataset("Y_valid",data=Y_valid) 
    dset_xtest=f.create_dataset("X_test",data=X_test) 
    f.flush() 
    f.close() 
    
    exit() 
    del X_train 
    del X_valid 
    del X_test 
    '''
    f=h5py.File('/srv/scratch/annashch/cs231n_project/best/imagenet.normalized.with.zeros.hdf5','r') 
    X_valid=np.asarray(f['X_valid'])
    Y_valid=np.asarray(f['Y_valid']) 
    batch_size=128; 
    train_size=f['X_train'].shape[0]
    print "making generator!" 
    train_generator=create_generator(batch_size,f); 
    print "made generator!" 
    model=VGG_16(weights_path='/srv/scratch/annashch/cs231n_project/best/weights.with.zeros.hdf5')
    #opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    opt = nadam()#GD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
    print "compiled model!" 
    #do some training! 
    print "compilation finished, fitting model" 
    model.fit_generator(train_generator,samples_per_epoch=train_size,nb_epoch=40,validation_data=tuple([X_valid,Y_valid]),verbose=1)#,show_accuracy=True)
    model.save_weights("/srv/scratch/annashch/cs231n_project/best/weights.with.zeros.nadam.hdf5",overwrite=True) 
    outf_model=open('model.with.zeros.nadam.Yaml','w') 
    outf_model.write(model.to_yaml())     

    #train_predictions=model.predict(X_train) 
    #np.savetxt('training.predictions.txt',train_predictions,fmt='%i',delimiter='\t') 
    valid_predictions=model.predict(X_valid) 
    np.savetxt('/srv/scratch/annashch/cs231n_project/best/validation.predictions.nadam.txt',valid_predictions,fmt='%i',delimiter='\t') 

    #train_scores=model.evaluate(X_train,Y_train,batch_size,show_accuracy=True,verbose=1)
    #print "model training scores:"+str(train_scores) 
    valid_scores=model.evaluate(X_valid,Y_valid,batch_size,show_accuracy=True,verbose=1)
    print "validation scores:"+str(valid_scores)

if __name__=="__main__":
    main() 

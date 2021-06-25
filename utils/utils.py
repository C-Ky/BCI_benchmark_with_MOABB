#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3


# General imports
import os
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Machine learning imports
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

        
# Encapsulate classifier (keras neural network) into an estimator, necessary to fit input/output for evaluation module of MOABB
class Estimator(BaseEstimator, KerasClassifier):
    """
    Generic classifier class that uses a given model to handle estimator operations
    """
    def __init__(self, model, name, batch_size, k=5): # adjust the batch_size for your computer memory, GPU
        self.model = model
        self.batch_size = batch_size
        self.k = k
        self.name = name


    def fit(self, X, y):
        """
        Required by scikit-learn
        """
        N_tr, N_ch, T = X.shape #N_tr: nb of trials, N_ch: nb of channels, T: nb of samples
        if self.name=='tcnet':
            X = X[:,:,:].reshape(N_tr,1,N_ch,T)
        elif self.name=='lstm':
            X = X[:,:,:].reshape(N_tr,N_ch,T)
        else:
            X = X[:,:,:].reshape(N_tr,N_ch,T,1)
        """
        # necessary if using loss=categorical_crossentropy: convert y to one hot encoded vector
        for i in range(len(y)):
            if y[i] == 'tongue':
                y[i] = 3
            elif y[i] == 'feet':
                y[i] = 2
            elif y[i] == 'right_hand':
                y[i] = 1
            else:
                y[i] = 0
        y = y.astype(int)
        y = to_categorical(y)""" 
        self.model.fit(X, y, batch_size=self.batch_size, epochs=750, verbose=0)
        print("fitting done")
        return self


    def predict(self, X, y=None):
        """
        Required by scikit-learn
        """
        if len(X.shape)==3:
            N_tr, N_ch, T = X.shape
            if self.name=='tcnet':
                X = X[:,:,:].reshape(N_tr,1,N_ch,T)
            elif self.name=='lstm':
                X = X[:,:,:].reshape(N_tr,N_ch,T)
            else:
                X = X[:,:,:].reshape(N_tr,N_ch,T,1)
        p = self.model.predict(X).argmax(axis=-1) #returns array of predicted labels not as softmax
        print("predict done")
        return p
        
        
# Loads an already compiled model (should not be relevant)
def build_model(path):
    model = load_model(path)
    #model = load_model(path +'best.h5')
    for l in model.layers:
        l.trainable = False
    lr = 0.001
    model.compile(loss = 'categorical_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy'])
    return model


class Scaler( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self ):
        self.scalers = {}
        for j in range(22):
            self.scalers[j] = StandardScaler()
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        for j in range(22):
            self.scalers[j].fit(X[:,0 ,j, :])
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        for j in range(22):
            X[:,0,j,:] = self.scalers[j].transform(X[:,0 ,j, :])
        return X

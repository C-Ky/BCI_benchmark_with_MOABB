#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3


# General imports
import os
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
import seaborn as sns

# Machine learning imports
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
from keras.layers import Lambda
from keras.backend import argmax, cast

from sklearn.base import BaseEstimator, TransformerMixin #,ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, get_scorer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

import pandas as pd
from joblib import dump, load
import pickle


# MOABB imports
import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery


# Local imports
from utils.models.tcnet import EEGTCNet
from utils.local_paradigms import LeftRightImageryAccuracy


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")



def auc_score(y_true, y_pred):
    if len(np.unique(y_true[:,1])) == 1:
        return 0.5
    else:
        return roc_auc_score(y_true, y_pred)

"""
def auc(y_true, y_pred):
    return tf.py_func(auc1, (y_true, y_pred), tf.double)
"""


# Creates model for the TCNET
def model_tcnet(classes, channels, sp, epochs, loss, opt, met):
    F1 = 8
    KE = 32
    KT = 4
    L = 2
    FT = 12
    pe = 0.2
    pt = 0.3
    model = EEGTCNet(nb_classes = classes,Chans=channels, Samples=sp, layers=L, kernel_s=KT,filt=FT, dropout=pt, activation='elu', F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    model.compile(loss=loss, optimizer=opt, metrics=met)
    return model


# Encapsulate classifier (keras neural network) into an estimator, necessary to fit input/output for evaluation module of MOABB
class Estimator(BaseEstimator, KerasClassifier):
    """
    Generic classifier class that uses a given model to handle estimator operations
    """
    def __init__(self, model, batch_size, k=5): # adjust the batch_size for your computer memory, GPU
        self.model = model
        self.batch_size = batch_size
        self.k = k


    def fit(self, X, y):
        """
        Required by scikit-learn
        """
        N_tr, N_ch, T = X.shape
        X = X[:,:,:].reshape(N_tr,1,N_ch,T)
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
        p = self.model.predict(X).argmax(axis=-1) #returns array of predicted labels not as softmax
        print("predict done")
        return p


#############################################################################
# MOABB application

## Parameters
classes = 4 #2 #4
channels = 22
samples = 1001
batch_size = 64
epochs = 750
lr = 0.01
loss = 'sparse_categorical_crossentropy' #categorical_crossentropy
opt = Adam(lr=lr)
met = ['accuracy'] #auc_score(for roc_auc metrics used for two classes in MOABB) #accuracy

## Making pipelines
print("Making pipelines")
pipelines={}
clf = model_tcnet(classes, channels, samples, epochs, loss, opt, met)
pipe = make_pipeline(Estimator(clf, 64))
pipelines['tcnet'] = pipe


## Specifying datasets, paradigm and evaluation
print("Specifying datasets, paradigms and evaluation")
datasets = [BNCI2014001()]
paradigm = LeftRightImageryAccuracy() #2 classes (right and left hands) #MotorImagery(events=["left_hand", "right_hand", "feet", "tongue"], n_classes=4) #LeftRightImagery()
evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True)

## Getting and saving results
print("Calculating results")
results = evaluation.process(pipelines)
if not os.path.exists("./results"):
    os.mkdir("./results")
results.to_csv("./results/results_tcnet_moabb.csv")
results = pd.read_csv("./results/results_tcnet_moabb.csv")


##############################################################################
# Plotting Results
#
# The following plot shows a comparison of the three classification pipelines
# for each subject of each dataset.

print("Plotting results")
results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
g = sns.catplot(
    kind="bar",
    x="score",
    y="subj",
    hue="pipeline",
    col="dataset",
    height=12,
    aspect=0.5,
    data=results,
    orient="h",
    palette="viridis",
)
plt.show()

